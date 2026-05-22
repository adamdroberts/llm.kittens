# SM120 Round Manifest

- run label: `codex_sm120_disable_cublas_bwd_tk_dgelu_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_disable_cublas_bwd_tk_dgelu_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_disable_cublas_bwd_tk_dgelu_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `645`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2190568` | `daf83b1128c4d9b22198e2853b2af62a6a0bb9fb2232d5a620bf79f0233e3391` |
| `test_attention` | `True` | `1800528` | `2821f98056d80125fe548517048bdcb7e61a53ace0d60da37b51ed83b20d9271` |
| `test_layernorm` | `True` | `1278296` | `bcd45bb3bf787c28969629ed8b184e3f5cfff0ca510d0033c57e5b8c571a2001` |
| `test_bias` | `True` | `2052256` | `a98c3c0bd4ddcd447fb73cf532d90e79918df3bf56070d8daf65369869b9fec2` |
| `test_gelu` | `True` | `1179912` | `77c3d8c361fcb19a1ff9ac244a932ec003e8877a22397bce57dcec011454327a` |
| `test_fused_classifier` | `True` | `1208704` | `0980f53fffdf30df9f0ed2f824a0d505a7c2f21f8b3885bbd67a326acc6bdd49` |
| `test_encoder` | `True` | `1210168` | `986eed79c9ea152670bf97031bd4e2f8ea70c3f63f08e59c2fca3459ba01392f` |
| `test_adamw` | `True` | `1183768` | `bacf83a1a75ad44c0a9acce73a6657a0fb0094bfced00758f908c63dd545a991` |
| `test_global_norm` | `True` | `1179464` | `793ba2aca36cb6d016cb926580cdae5e320092e7cbe71ac0f3fc957fbfc95280` |
| `bench_sm120_matmul` | `True` | `2369344` | `df2dd2af93cc088f8655fdd4c91a3ec392ddadd24fe3b35e6f106f9d6bfcc736` |
| `bench_sm120_attention` | `True` | `1768800` | `5df93e1d84cf2d9007654c419701fa25c127b9d44b2ecf33bcd466be5c087834` |
| `bench_sm120_layernorm` | `True` | `1274232` | `e21eef51e3d84bca866d84919d09a8ae8cdd667a5e5fcd376a61e0b1c84babe3` |
| `bench_sm120_runtime` | `True` | `2234304` | `43f59d24cedc2ed1ecbdceab5558aee1d497e9d841a9ad96932ada8a859b97d7` |
| `train_gpt2cu` | `True` | `3104304` | `64551d3f333722d8bda80fcc900a70c330a8fbdadc9a40d34639f32e7eaa178e` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
