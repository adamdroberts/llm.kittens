# SM120 Round Manifest

- run label: `codex_sm120_runtime_grad_zero_zero0_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_zero0_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_runtime_grad_zero_zero0_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `673`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `0e5b6c2e9ee9eee35b0f532cf102d4e612df91f5e8b1497cb911a8ceb19346a5` |
| `test_attention` | `True` | `1800528` | `5c2e42cc46cc2fc4581df2458416873f84410068f0a11e0fc806d30310b780f1` |
| `test_layernorm` | `True` | `1278296` | `8906e3e3fba5486e3534edb53733ca533f9c6a21d04dc44928625c4bd09d9482` |
| `test_bias` | `True` | `2089120` | `a2744ae9a6e55fcb97bed04cb0dbdb86bdd9b60f9b3f6dc54551a9ba6a67ad9a` |
| `test_gelu` | `True` | `1179912` | `18baa2372d6300cc9b5e585c9c633e176fe78aeab5a04a504dd56f665b8e59c6` |
| `test_fused_classifier` | `True` | `1208704` | `44cae02812656be8e5ccc7d5531cd1a11c7282a6c6e88e1b43c3013fbafd491a` |
| `test_encoder` | `True` | `1210168` | `ea51d499c2c064e298bed7d5a9fa36a9f0241df27f6cd1e196bb4e51765141eb` |
| `test_adamw` | `True` | `1183768` | `c8bcea7192ccec16ee03ad9dfe471daf41b2356922ec399cd0eaf927e1caa16d` |
| `test_global_norm` | `True` | `1179464` | `91488e01d018c681e5236da1854f41e3134b9f2a8fdeda6da1f4d9536eaae22e` |
| `bench_sm120_matmul` | `True` | `2410304` | `a3de67c8c693b240d1b8f59a14289c49d1f1f48ae7ffad531dbd4b628fbe2cd2` |
| `bench_sm120_attention` | `True` | `1768800` | `98accfb6d37ea845d1aa65fdd5a484488d8862eea0ba26b68ee66fa0900a3bfb` |
| `bench_sm120_layernorm` | `True` | `1274232` | `94e7d9a3cf82431bb93ec3df019ca3cd9972656020708a81e004970166aa7b4c` |
| `bench_sm120_runtime` | `True` | `2271168` | `ec9cc672d8ba0fb1ed948ff5896459f01695b0f17cfc51441d0b0a40b9d2ba5d` |
| `train_gpt2cu` | `True` | `3105552` | `20bef188744eb105e71c7ef63b8caa5031a6038394a6e1da66c9dd2914d6a560` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
