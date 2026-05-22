# SM120 Round Manifest

- run label: `codex_sm120_round_libtorch_grad_zero_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_libtorch_grad_zero_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_libtorch_grad_zero_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- git commit: `0f21747`
- changed paths: `502`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2185024` | `89d7884e7f4fb9bca8534700bf46f8937737a3b9b2af41f24bed23218ab87343` |
| `test_attention` | `True` | `1807832` | `127413c72009af2e07fa4be12be68c8d36f82729eb81d721483502a19fc26675` |
| `test_layernorm` | `True` | `1285600` | `f9dbbcfe9d6c8ed48393670691c9976c79e912bc3d0860816b41d6999b19398c` |
| `test_bias` | `True` | `2096424` | `2a934ccb14b911c08ce42dea6ed1a7c3c0264bb59c7dd482bf038eff44697aa7` |
| `test_gelu` | `True` | `1183120` | `62585ae67bf4716e18ee68dcef37d73d8d6a9d459127bc0a3b7ac499f72d843e` |
| `test_fused_classifier` | `True` | `1207816` | `01ed71d728e98e8671a26ab492cae373c7f908c345f2bbc31c30d4355b5b0a77` |
| `test_encoder` | `True` | `1213376` | `9110aade4595424a211f82978157055e2f57b4ddc7a2f928011d653cbd81c155` |
| `test_adamw` | `True` | `1186288` | `2ecd1b9edae95c4f51975bd3a265dbc7a19c63226748aee7d5cdeaf64b665288` |
| `test_global_norm` | `True` | `1186760` | `7bc1c74dae19884acddd8f5b3cc65e5c22319a7a3934ce2ad1b9df0ac7b66376` |
| `bench_sm120_matmul` | `True` | `2421704` | `4f32323556edc7f933a90c57ab4ffefd9d66b899531e1e30820572feeca65f9e` |
| `bench_sm120_attention` | `True` | `1776104` | `c4a1c172a0e51d2efd91ad952a7a2ff6f47671202f3da1ae0348ca137ee09866` |
| `bench_sm120_layernorm` | `True` | `1281536` | `ff0887aeb66582f22278ffe414d48c241cf54adda63ee1499d81b6ae7a7d2666` |
| `bench_sm120_runtime` | `True` | `2269656` | `b893d009993c6fcea741c687af8847b631c8db2984650180037d68515e8850f3` |
| `train_gpt2cu` | `True` | `3111984` | `798c4f73ae43791505f3bf09ad7f6e8ab78087a56e0ae239c50a428973effbf3` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
