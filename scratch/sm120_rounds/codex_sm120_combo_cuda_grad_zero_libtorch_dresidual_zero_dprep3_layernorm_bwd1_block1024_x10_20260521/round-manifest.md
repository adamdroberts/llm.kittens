# SM120 Round Manifest

- run label: `codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `595`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `8ae7ea2e92f78eca1b1c36da3f628340a9e26c9cfa3663ff23450e4c8f12a492` |
| `test_attention` | `True` | `1800528` | `a0bded0cf96124399f42473ad7ff33e4d5aa441c408b009422b61fb7d0fa2ff6` |
| `test_layernorm` | `True` | `1278296` | `af3faea42e62ea5da623e2dd38ffcf233dbeec026e2fe5e45cbf3bb50938f0e7` |
| `test_bias` | `True` | `2089120` | `677e75623bde2a0fbdc89f2a5dc765f7dfbe753fc0d67cbad2f1344b7a3e9118` |
| `test_gelu` | `True` | `1179912` | `d3f2b55c9b538ff091d663fd2cb5c3c4019e62b47c1d0507774a5f2aa14b57b6` |
| `test_fused_classifier` | `True` | `1208704` | `23634674a4764032461625f4f53a9d993687b71af061af528f1a149049e1e564` |
| `test_encoder` | `True` | `1210168` | `680ce9f91cce7d30b0d10051f26965e304e1121174213ad191d5ce94f51995a8` |
| `test_adamw` | `True` | `1183768` | `1180dddc267db71eb3eb4d2e66227e13bf34f2c725fad53e410013b49879de42` |
| `test_global_norm` | `True` | `1179464` | `331c52ae82edd5282d8f5103ad3978f52b25e71133591e31ce2e552c48f57649` |
| `bench_sm120_matmul` | `True` | `2410304` | `05cd807151293665b9c4af83774ac4568d31977753e9b03139440f898d24df38` |
| `bench_sm120_attention` | `True` | `1768800` | `0ea0fbc73374698e6f517e3d465ede528b2f29ce068f28e6ad27e135596cbb7c` |
| `bench_sm120_layernorm` | `True` | `1274232` | `4025db771cd58a805fe92e921cd65bea8ae3e0039c293b2180281eb66ae44bdf` |
| `bench_sm120_runtime` | `True` | `2271168` | `b4c315621709fc93502f2d9af5eb6e294183d858130b2d57cb7044e3dd04b1a3` |
| `train_gpt2cu` | `True` | `3105552` | `a8cd1f3cd70111769cf3c979550bedb6cfc90593e8b247c6ad7ef0579668a125` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
