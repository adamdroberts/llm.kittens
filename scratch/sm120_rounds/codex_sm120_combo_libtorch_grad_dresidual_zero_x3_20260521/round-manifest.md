# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_grad_dresidual_zero_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_dresidual_zero_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_grad_dresidual_zero_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `518`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `5749bd46727ce36e53dcbf9f0bbc978eade377fcfcfab0eb285be29c554165a8` |
| `test_attention` | `True` | `1800528` | `de7bb19c580b5afea0e948ebb6fd089e4c5c624301d106efe10a98b52835a605` |
| `test_layernorm` | `True` | `1278296` | `b3d588061fa404a7f83ffde8251b12c77104dd64703cdabbaaaf738d8451df0b` |
| `test_bias` | `True` | `2089120` | `c4820a58b2c2ea902980c0a73314410d73629fa714558fb548516c961395d502` |
| `test_gelu` | `True` | `1179912` | `35e6d47fd37305fe5d2821d135d5065efee6779fc8a573ac1847b3f0f83b3ef0` |
| `test_fused_classifier` | `True` | `1208704` | `fef99db170656e2f6e13e360f7491b4d652d0bee84f28f2112380760cc286aeb` |
| `test_encoder` | `True` | `1210168` | `804e19b65e37b904d46960088c3c69e99a148c03ed4843a2eb108d256a2cca34` |
| `test_adamw` | `True` | `1178984` | `e5bb66c872a3e0601b6dcea37e60cfc9cdf51a6e561b09cefd67a23fe9ce5eb7` |
| `test_global_norm` | `True` | `1179464` | `1e6316f3f095aeb614facd5ccabb96748e59f408a07d112bf5e2596c042dc337` |
| `bench_sm120_matmul` | `True` | `2410304` | `b36034b4cf9d7dd1c58761f9e7ccd43cc04245a2558f9058c398cc400ef9c762` |
| `bench_sm120_attention` | `True` | `1768800` | `6de02ae9ea7b90b0d672720a1735e2c747520127b7a11c9cbd2041f1cdbd2e9e` |
| `bench_sm120_layernorm` | `True` | `1274232` | `7852e03a69ef58dda17be2a52cd9a2db3464c752fdebd8806402b3cdd517d680` |
| `bench_sm120_runtime` | `True` | `2270544` | `fcde9c543ea112cb27e62dc582d66b5caf66d40246553a3308626b332c32b575` |
| `train_gpt2cu` | `True` | `3100672` | `d53f2a833c99ccfc98eaa352fc8e45b4eb6b728d84f3ffd50d01e19ceeed0784` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
