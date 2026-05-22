# SM120 Round Manifest

- run label: `codex_sm120_round_current_mix_final_smoke_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_current_mix_final_smoke_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_current_mix_final_smoke_20260521`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `479`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2137216` | `e7a99451b9fb1bcdade575ab0be67d5cb841d6e28c058266aab82072db1aca82` |
| `test_attention` | `True` | `1760032` | `8e50c85dabe9ee62ff16dda62fa96e01194808a0225d1613805321e6d1b2eb2a` |
| `test_layernorm` | `True` | `1237784` | `463403df6a9823f088d6ee79f944da6692080793713f621c4d0f478e671cf33f` |
| `test_bias` | `True` | `2048616` | `0e26b1491d24153247d03233e3de1360f8783498b539af5e649178836d69a322` |
| `test_gelu` | `True` | `1139336` | `2e631731c2b82c3e60de12e5181349205ba038329b5ae612c7bd5bba3b13d28d` |
| `test_fused_classifier` | `True` | `1164032` | `1f12e3584ee1109cacdc7a5c7cefaf6605275c777e708d3acf25bfff5dd1b403` |
| `test_encoder` | `True` | `1165512` | `f87c9a608ad17495208f26c2bf0332dfaf19fb963dea5eb0bd67a341cf5240e9` |
| `test_adamw` | `True` | `1138408` | `cb8bd44adfdb6eb30521dde624e16df744c9275b0b156ae56fc260f8f366188b` |
| `test_global_norm` | `True` | `1138880` | `f3a01cccb92850873d49f9cc31d1c0cb7fe78413584bdb6b3a2c586913f556da` |
| `bench_sm120_matmul` | `True` | `2373912` | `421d9812a5ddbace1dab48f001057c10ebe1c40d1100795b023cdcc4dbb2ccce` |
| `bench_sm120_attention` | `True` | `1731352` | `f0f232c49b371f3cc4b5cac014d7ebce4adf5eaaed85a0c861f75afe2463568b` |
| `bench_sm120_layernorm` | `True` | `1233728` | `43e767444955876e91c2ae84e90562fa55addb38c281b902f2759be784e7ad75` |
| `bench_sm120_runtime` | `True` | `2217576` | `1ea8a6da465961eb0a5fbf2b6686088fe73214e977e88a0e7342a650c25e09c4` |
| `train_gpt2cu` | `True` | `3045944` | `54b9d6308c50d5b7dcf883c182fcb6cf8963a161c302d2a8ae5525236c92da13` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
