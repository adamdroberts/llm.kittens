# SM120 Round Manifest

- run label: `codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_confirm_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_confirm_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_confirm_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `604`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `f1186ca9b800ce170b368a17b1a814411e2a2ff5e1c398453d86ffe34b73e911` |
| `test_attention` | `True` | `1800528` | `d52fcfd146e25caa838cf7b38236dbbe4f60c78851f7f820b10c661293cafcaa` |
| `test_layernorm` | `True` | `1278296` | `4b629ca298878c933f06199befaf222637c461416a97ec0d920fd849d2b1227e` |
| `test_bias` | `True` | `2089120` | `988b72de21d4533ce2e956b0b47ebaea1cbcc5b498da66bb2701376e83eccfba` |
| `test_gelu` | `True` | `1179912` | `f1081015ef9a2218c689d27593d36f39813ff97af787179b2f5df7e77ecade7c` |
| `test_fused_classifier` | `True` | `1208704` | `5c5bcfd0ca7c7d0acd0571ef254954c824c5e315e51a884ea0aab72ca0b1b384` |
| `test_encoder` | `True` | `1210168` | `d6687dad3a0499a87d12d595565bc9ef9663401187fef3678c51cd3a47847376` |
| `test_adamw` | `True` | `1183768` | `034d2dc23b1ba8435d6d4a2793cbae4864857574d233525c73b1fb8b40d5228d` |
| `test_global_norm` | `True` | `1179464` | `33dd4de693fdf9bb4a1844c9fdfa56f5d79b7d55ebd81cd4c3d46c0c6ad92098` |
| `bench_sm120_matmul` | `True` | `2410304` | `eae19487807751236bf91e4726672488560e5c99251251f77e98aca847a95c38` |
| `bench_sm120_attention` | `True` | `1768800` | `0622462f8f23a41b60334f7f7df9ab96749c2604ea374f3f98e38d8a2e1bc520` |
| `bench_sm120_layernorm` | `True` | `1274232` | `3e036c74819a2ee0d6354af9c690c7eef37b58f10914d57e12fdbd3967096557` |
| `bench_sm120_runtime` | `True` | `2271168` | `a3b2998d6ecb2ef501ecd40479ac0402e2ba6724827907729e6e113d560b450c` |
| `train_gpt2cu` | `True` | `3105552` | `99d2163520d1a425de0ec26528942a54984191fa4d41fb14672fc6014833a8ef` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
