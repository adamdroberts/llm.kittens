# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_maxconn1_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_maxconn1_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_maxconn1_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `576`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2190792` | `228030e130f2aee69a100e8f6ee9d51b53f2d68b8fc4432d357d393bd649e310` |
| `test_attention` | `True` | `1800528` | `6793f216da4fe3e6c7eee3d3053dc263b1489a6010574bbd19e64828ba82f58f` |
| `test_layernorm` | `True` | `1278296` | `515dd50d30d276df6b8ba412a272795e4dab7b92992f85a4da3840c1206f60ef` |
| `test_bias` | `True` | `2052256` | `3d4a47bdaec5d6862dc751cdeb42cbb41dfb82ce09796d905bd05aa5d09795c9` |
| `test_gelu` | `True` | `1179912` | `5b04f6e8b0bf07fd0b518e6898ddd8f82577e193dd63afcb72dd5bf5e151d49e` |
| `test_fused_classifier` | `True` | `1208704` | `002beb78158b58798c451afac92ee93426f8fccdeefd2c669309bfdd8a168c9f` |
| `test_encoder` | `True` | `1210168` | `6ff55eb21c2553e72b2d61e7dbdebcb66beeaa2c73ff4d238a4858cd42b8297d` |
| `test_adamw` | `True` | `1183768` | `ea19549399809bd066259f01bc714e9a2bc84624cbac83289167251ed5c8cc6c` |
| `test_global_norm` | `True` | `1179464` | `4ca21df3f6fd676b0d9730c6917c737c3e58c0009e988d21c6c66d4a657d4231` |
| `bench_sm120_matmul` | `True` | `2369344` | `9b36b030e4f0a9904e19a69eda68e3b7d25712d66ecd2231311ba1c888123f11` |
| `bench_sm120_attention` | `True` | `1768800` | `10b3bbefe81cb9dd747d810afb9046f5b32b90eb89f3d349c62eb8ce5736bfae` |
| `bench_sm120_layernorm` | `True` | `1274232` | `3ccabce5b8c079e19a1dddaa94ea5d8501a26cf4f85c50992fdc8c764782e7da` |
| `bench_sm120_runtime` | `True` | `2234304` | `0958a866bca1f98b78cc609ea31299fc9987d10167b31efaa9eca0b29907a1cd` |
| `train_gpt2cu` | `True` | `3107416` | `def0bf0462cb7c09620914ae53258b8768146ddbb937acd5f3d0cce9d46a658f` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
