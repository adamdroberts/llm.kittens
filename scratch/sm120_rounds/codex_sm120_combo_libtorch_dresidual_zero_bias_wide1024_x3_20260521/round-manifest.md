# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_dresidual_zero_bias_wide1024_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_bias_wide1024_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_bias_wide1024_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `556`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `a272458116fb15cbca4838b1e18c5c5782a65082ce70e1282fbef896c1b74a77` |
| `test_attention` | `True` | `1800528` | `ef9b3059de5658ad6a2921229e0ade1a534189615ab03001e73400c79b7cbea6` |
| `test_layernorm` | `True` | `1278296` | `514477e37463c930bcb5612985a143637797c462551d8a2c69ab4ac23e71de09` |
| `test_bias` | `True` | `2089120` | `3fa5e7e28807381aa1f928421f3f3fe7dc0d7250df9a88910c6686cf0b240dbe` |
| `test_gelu` | `True` | `1179912` | `ece462122fa4ae94d09ea90ff99006ab235769e62eb8788cba239a2420387c90` |
| `test_fused_classifier` | `True` | `1208704` | `0403efe89f6fbc2c630ae9a7cf598e06d8b40ae69b9545d61d3ae287646cb401` |
| `test_encoder` | `True` | `1210168` | `96ca4f813cad9680f7701589c7b0c8ceb744ba535d12cff65fef27aef3666a1e` |
| `test_adamw` | `True` | `1183768` | `b57b030ff3f5de196145746e5c8fa62a59ac1bcedc5842b3f0c537a2dd63261d` |
| `test_global_norm` | `True` | `1179464` | `d05383aee2ba3a225101015e9eeca55869505f11b6dc15b82a69027186098da1` |
| `bench_sm120_matmul` | `True` | `2410304` | `77a553f0dcdd318b972f886fa8e945836263b24c49c2ace49805fff7b2cae22a` |
| `bench_sm120_attention` | `True` | `1768800` | `00ed96fa3ec386298ae8d7e56f48c58dca5221387094bc9ce91bad3d05867d43` |
| `bench_sm120_layernorm` | `True` | `1274232` | `b0a3b8d7d3f9436234fc6df5736470f51392d6614cb1bcb8634788a94c780114` |
| `bench_sm120_runtime` | `True` | `2271120` | `6380e5131de4127999889d0a6e3cd6b498a13166cdb5f3b6f0628aba68418e2f` |
| `train_gpt2cu` | `True` | `3105552` | `3357a7c5346e24169302357a1351c602ab00023182b30e1eba312a040a0f92ab` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
