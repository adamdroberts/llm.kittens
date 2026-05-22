# SM120 Round Manifest

- run label: `codex_sm120_selected_restore_after_libtorch_grad_current_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_selected_restore_after_libtorch_grad_current_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_selected_restore_after_libtorch_grad_current_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `682`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `8b4832fbab0e47bd9d757374d8ea44111c97cc08d9b9145bdd7fc34f97da945d` |
| `test_attention` | `True` | `1800528` | `c473cd2ca2ea72bf0750aa12bfb7480ae13c627cb219ee13783371dcd5daf031` |
| `test_layernorm` | `True` | `1278296` | `31e062569493eecb476d803d4d590adcf725a644eabd53538da860db5dc821ae` |
| `test_bias` | `True` | `2089120` | `1941f58c0facebb177bd22b0ac911c99a77d584b6edf93645efa45f0d0d3d352` |
| `test_gelu` | `True` | `1179912` | `638ed8b4003e1a053951df971faea449d845aa688506db146f3e1b457e6ead8d` |
| `test_fused_classifier` | `True` | `1208704` | `22d12d7a1689e7b329fd25b4884df5405193ed83372e80ee9a302aa4acc4c710` |
| `test_encoder` | `True` | `1210168` | `2ec3febefcc7afd86189891e522aa4ebc869cf3bee95905647d2cc25aa983f3e` |
| `test_adamw` | `True` | `1183768` | `4c8e8efd7b318e9d8d515d7c0456f9ec7a17cda3150062e4aeb3db0cf46ee030` |
| `test_global_norm` | `True` | `1179464` | `d6e89287b39aa5c3dacd5b782028ffac0e9378fde2bae1eed3cfbe64d049544d` |
| `bench_sm120_matmul` | `True` | `2410304` | `3d145fee4c628f575e2aeeccdb5f302364a6369b8ce9656e7d848fcacfba8da0` |
| `bench_sm120_attention` | `True` | `1768800` | `b5f34d10ecb568a22c7ae74cbf21c1619357e7909ee90c2263e86f643298b082` |
| `bench_sm120_layernorm` | `True` | `1274232` | `299f98389ea14acc65ee878fe0e787ecaec9bcf11f011a04e5e87e2b4872f99b` |
| `bench_sm120_runtime` | `True` | `2271168` | `1172a0ac158c23d445ac16b43910695a265503db8d95308c2b4009b8ec2ddc16` |
| `train_gpt2cu` | `True` | `3105552` | `c53b948f34fdc061a706a4e3f7f69f843cac31b2a30cbab41d929278ce303770` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
