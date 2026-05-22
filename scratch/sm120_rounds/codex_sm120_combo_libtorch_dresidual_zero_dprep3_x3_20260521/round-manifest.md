# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_dresidual_zero_dprep3_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_dprep3_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_dprep3_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `588`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `53a215f1572f98b32831de817b00ec46692d9787432e43b379cb7add21fbeedf` |
| `test_attention` | `True` | `1800528` | `a20e28bea3b228d392e863a9a9884ede1e77177a66d3c9e0c1267f8897e562fb` |
| `test_layernorm` | `True` | `1278296` | `75557dd60fb0d66244b981ac4899b654e78923f4c268d28935207e4929eb2d6c` |
| `test_bias` | `True` | `2089120` | `05c793a4000ec4935812f6cc6076bdb60bb0ff3e13ae7ff5b1a234d58796a404` |
| `test_gelu` | `True` | `1179912` | `0343aba2d53264a40f137dab39e95e3011f122fe33a2720a56f775b5c19f8885` |
| `test_fused_classifier` | `True` | `1208704` | `25d5c07471f6ff31ccc79b23c2b44cf6f755bc1f6fbda3960a39019421f0a8dc` |
| `test_encoder` | `True` | `1210168` | `744daa073fccda6711df8f54993aea31b7c7370bd428b2cf1ee18281ce92bb81` |
| `test_adamw` | `True` | `1183768` | `679b0a704d3d8e2512aacbea262137bc85701f4d3b4dd54570f3b55998743655` |
| `test_global_norm` | `True` | `1179464` | `4e0c62d93b070d6c7551d61f1a157de413854af9d58115654d08bac4e394b4aa` |
| `bench_sm120_matmul` | `True` | `2410304` | `b35abdf6a0f3d54f8600412c2393331459ba98139f4bde336366f840666fae40` |
| `bench_sm120_attention` | `True` | `1768800` | `968142d6a7de8fd6eaef77b36ca178f61b18cad3160ed2b98a8e4ad65a40daea` |
| `bench_sm120_layernorm` | `True` | `1274232` | `98a1c70446d4a5430b487935e996661c42601d95c373ed6be918e9650d824c23` |
| `bench_sm120_runtime` | `True` | `2271168` | `ab05cd689fdddac4f320732ef665c6612fec4bfc8559512eccf350f4323578bb` |
| `train_gpt2cu` | `True` | `3105552` | `0ed88ed4274deb9c41d8bacf9bb51b7ec9dbef651dd511ac7fff30f64aba7939` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
