# SM120 Round Manifest

- run label: `codex_sm120_precompute_grad_scale_maxconn8_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_precompute_grad_scale_maxconn8_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_precompute_grad_scale_maxconn8_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `656`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `9bca6847ec7cdfe473778a1826cc443724a456cd21d3d619c8867b5424f19e8d` |
| `test_attention` | `True` | `1800528` | `d24da453581bbf4bb841c7bba6634272417b536df9d4eb3ddee0fe8b5e3d1b92` |
| `test_layernorm` | `True` | `1278296` | `fcd91248dcc179b716fd2dda14430b09aef89fb7f1b2598fc1690150d53a9710` |
| `test_bias` | `True` | `2089120` | `02278dd13c70ae3e511dd5c273c13fe5fb13a300275625f052da3f1b6d570556` |
| `test_gelu` | `True` | `1179912` | `ecaecaf5d88d714cd095ac3606fece67bb9f9c16a3bd776ecdbb5f3d54eefcc4` |
| `test_fused_classifier` | `True` | `1208704` | `01fce97d89aa112084195f56048b67f1a3d4fa81ec66fb1029d07f0618be2fa5` |
| `test_encoder` | `True` | `1210168` | `ee1089b4c4da64059dd4ca9b05d87d53a1adfe1b9e87c5908511a76e6251c126` |
| `test_adamw` | `True` | `1183768` | `5a4fe04410be7d5f3158f0431ec06f958be16229e28825ba3713f35adb7c7446` |
| `test_global_norm` | `True` | `1179464` | `7b08864b20c17d1896fa1cb074e2a13e5ef6df4f292c8902f41128ad0f70e28b` |
| `bench_sm120_matmul` | `True` | `2410304` | `67c5d324631384f211fbb016b4cced7478f4872172e2289d8bd700ee3a0c7584` |
| `bench_sm120_attention` | `True` | `1768800` | `9e791528425a7cc305766497751c0574b2a064eb0345f0d1971462dbd44854a4` |
| `bench_sm120_layernorm` | `True` | `1274232` | `75fb6f76c6c7708bcf5d30a08f66f776b2e6a9c81a9a85d3ab4bfac4720b1c2e` |
| `bench_sm120_runtime` | `True` | `2271168` | `589918bc3dfdf1e227b8ea8014e91a21f62632eff0dba9e04354ec84c05aa477` |
| `train_gpt2cu` | `True` | `3119216` | `7ad1d08c7a68f5f5e871b0f9bf4d249d12ffdaeb87d615275abbfb4f60239a30` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
