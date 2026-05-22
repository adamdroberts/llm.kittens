# SM120 Round Manifest

- run label: `codex_sm120_promoted_cublas_dinp_tk_dgelu_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_tk_dgelu_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_cublas_dinp_tk_dgelu_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `618`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2190792` | `9128daadc9bbd736256664db7992733635b62344b5b73999627a61981d4abbb2` |
| `test_attention` | `True` | `1800528` | `6cf4e00cb4ff221c6cb22180f31085c3528845988a9d101652886a64cc72fb66` |
| `test_layernorm` | `True` | `1278296` | `5bcf633ae6534b116aed4b6d903c0319aee52d4f2d63d4f4f98fc9865f0bf712` |
| `test_bias` | `True` | `2052256` | `98791b8f18d31fa7ce2553655830675ae4764841768a0bd0e2c29c648d9c4451` |
| `test_gelu` | `True` | `1179912` | `798c8a9a98e657d83d85f5c38028d708535812b2bbffe123880950c6322b40aa` |
| `test_fused_classifier` | `True` | `1208704` | `f0278b9b20bb2fe1c83e633f1d06b03a99482d1d8c5570b550180f66ae257b80` |
| `test_encoder` | `True` | `1210168` | `6c538e88d649bdf096c93c93b7a0975de2c6a5a90e8cf7addf0a202b4334af28` |
| `test_adamw` | `True` | `1183768` | `1398e345a0f559f2180676f29d6e73b5ff6c76c40508cafdaede50eba12d1535` |
| `test_global_norm` | `True` | `1179464` | `61f0d3dfb9d42d8a1407f03f2eba0063c28cd4d3b5763ee011037e02f7735211` |
| `bench_sm120_matmul` | `True` | `2369344` | `5ef4063ec29c1ddf3cd87b7c29d7f781a9ce4706150de7fa248a165f782299aa` |
| `bench_sm120_attention` | `True` | `1768800` | `b1ce5bff7f50eabef1501779ff2afa349f3c2dc141463084e86f0a2884caedbb` |
| `bench_sm120_layernorm` | `True` | `1274232` | `690e0b065279b258ec034ebf9c09c473d78df77c8dac290998b19763780e723f` |
| `bench_sm120_runtime` | `True` | `2234304` | `ed035c6b50ba9845ee18c3420bc2b071d7bc5c375f65a409e2b31329e859833b` |
| `train_gpt2cu` | `True` | `3107416` | `463c358cd928c1fa1ecb60f56815f9a18043fa52a9c63ede819a5a5e156740ed` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
