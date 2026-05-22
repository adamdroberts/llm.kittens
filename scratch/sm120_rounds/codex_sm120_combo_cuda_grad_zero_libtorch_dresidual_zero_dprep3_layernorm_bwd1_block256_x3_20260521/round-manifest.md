# SM120 Round Manifest

- run label: `codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `600`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `7d2ce375ced2a69add51ae0ed3cd9425277aa5ef3e5a55e39e69a2be7e597264` |
| `test_attention` | `True` | `1800528` | `7b6f47e042d86fd6c858be59cef0d6588023882057205aa4d60d183d3c3423a5` |
| `test_layernorm` | `True` | `1278296` | `8c5c5e662f9355ddcb45b900ee75547a11dc7042abb7711825a039936f3ca1bc` |
| `test_bias` | `True` | `2089120` | `f28123fd590ee2b9b63b1c13d31530fb383c48ab69b59e2b62a1c78e7ffee961` |
| `test_gelu` | `True` | `1179912` | `678d30d9a906586e790b1f31620fd81f09faa4ab0f5645b75b8426c574c77caa` |
| `test_fused_classifier` | `True` | `1208704` | `5c29cdb2edd59717c05c2e78798551dce87950ec1ff0afec4e2580f804984857` |
| `test_encoder` | `True` | `1210168` | `ab9743615fc9dfc7ec707d8a89802a887dcd15f15d822b446d55953ab6f69f01` |
| `test_adamw` | `True` | `1183768` | `be51fc7677bd1fb7e3211e9d5629f2d9bcb9d4d348e92e92a46e4eb556050f3c` |
| `test_global_norm` | `True` | `1179464` | `6ea10bf1b47dbb87dbd44cb42a2e40e50151b17e62c7a349a86020d868d0acd4` |
| `bench_sm120_matmul` | `True` | `2410304` | `d1f08f0b63c3c41f116e94be222f2942a62b5b832605d2f89d50f625bb97afc0` |
| `bench_sm120_attention` | `True` | `1768800` | `9515ab93ea02455509b4e73f8f772e9cba616112b373d39422dd12df34a4285d` |
| `bench_sm120_layernorm` | `True` | `1274232` | `0abead6fe19e62f567380a66dee51c6e919039c26c7d7decffc6a2bb51caa49d` |
| `bench_sm120_runtime` | `True` | `2271168` | `6c9116feda4612f79840a562eb33954eb8b862bd603c1cae06116f8716f10f3c` |
| `train_gpt2cu` | `True` | `3105552` | `42e4aafa0d36ed6456f74876591eebe6086df5ee79695c9f24f3ea2e504963a0` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
