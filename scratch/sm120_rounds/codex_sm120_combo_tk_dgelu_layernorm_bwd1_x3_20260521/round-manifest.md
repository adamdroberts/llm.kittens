# SM120 Round Manifest

- run label: `codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `517`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2154392` | `158ef32be13cc58d5dffcef6de203ad9bd885d641e35d13691065a398e27b4b3` |
| `test_attention` | `True` | `1760032` | `2dcbc399c2b3e853546ebfc9142c2bf81086a963cf5152f993c3ce81fed8a5a0` |
| `test_layernorm` | `True` | `1237784` | `d62ddc31f22f829d572bd27ef8a066b11373853f43e69005ce8733cbbefdcfea` |
| `test_bias` | `True` | `2007656` | `c44edfcbd4ece31d0f8be932107ced6c60cbca69942391b6569798ad794f6d11` |
| `test_gelu` | `True` | `1139336` | `bb9bc5d06a255e4148868359014c58b1f87517757a9038830794381c23128aea` |
| `test_fused_classifier` | `True` | `1164032` | `3873f563fe99031d0be0444c8d97858ff6fab1edeac87e34d33a8ec89420fee7` |
| `test_encoder` | `True` | `1165512` | `c5902c4abb8dc11b75349034179d9b2a980eda6f3fc91d12df1fe61c108964a5` |
| `test_adamw` | `True` | `1138408` | `f8eb1d25921597bbc60c3ff5d39126bcb61f8972d8773fc8d64743763adca649` |
| `test_global_norm` | `True` | `1138880` | `2cb1a8abed421b056405a29695da9059e7b8e828391fb04f4274c7d126e4e47c` |
| `bench_sm120_matmul` | `True` | `2332952` | `ed17d4bb9ec74d6a77439a272bd04d2a49d5f5d1f72480272b48d7eaa764d2a1` |
| `bench_sm120_attention` | `True` | `1728312` | `00d62098de3817df6e4fe5c144cb727df16d3df48c9256ff935190dd379ca6d2` |
| `bench_sm120_layernorm` | `True` | `1233728` | `325ab0f7bcecb41210a3d851886a1cd8bf7aa84505c8e744bc63048f9e366293` |
| `bench_sm120_runtime` | `True` | `2185000` | `36e9032c6108716a60aeaa6dd8d27ab3187a59d3714f0a12d5f7e1b39b36f709` |
| `train_gpt2cu` | `True` | `3054992` | `b86d42f6f4a071f3f347311a17e52b3bcdba0da688596d13e7890bf97be31730` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
