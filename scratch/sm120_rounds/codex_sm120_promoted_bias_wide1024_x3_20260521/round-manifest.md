# SM120 Round Manifest

- run label: `codex_sm120_promoted_bias_wide1024_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_bias_wide1024_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_bias_wide1024_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `606`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `d3ac271257259cdf822b8e5f47e05c6a8edde622da27723972dbdccc4b497861` |
| `test_attention` | `True` | `1800528` | `d3ff8addc7717eb703bd6c3532e04c9d2eb32d6678a2ebb80c91d409edeb1320` |
| `test_layernorm` | `True` | `1278296` | `289055aa1c49a13cd6b581670cbfdc418e2ec9186feef2f3170864621db4e258` |
| `test_bias` | `True` | `2089120` | `1e471f0b652a47f6727d2109480b612a035f2f0654f85b5be3f1fb01ebd68cf8` |
| `test_gelu` | `True` | `1179912` | `9d070f0468f93c50e6909bc2cf15e498faebe76d86bdd3c4c810a4e854a7dbf4` |
| `test_fused_classifier` | `True` | `1208704` | `de6eb909584cdc6ecd32178aa129d2c4aa85b7e040c920576b6ab95f4f51de10` |
| `test_encoder` | `True` | `1210168` | `8ca713498ccafdaa7db606077ef36186e76142515a38f567de58e7b1fdc35cfb` |
| `test_adamw` | `True` | `1183768` | `3279821064c48879491984fe9afc3f67375c2a5dddbd04eedccd96330eee6af9` |
| `test_global_norm` | `True` | `1179464` | `24d63b50da9ef7c762b7e5ba58506fa82005bf2acd7e635613bdb7921875df98` |
| `bench_sm120_matmul` | `True` | `2410304` | `8e2fe0ec4d768469989734daf1c2edd1fe47612a1edb9d0dd0d0850f706cd5c6` |
| `bench_sm120_attention` | `True` | `1768800` | `3ecc695a851b6d47167dd92b249f09a0a3c5731dacf95744e33efbb9d5b3d74a` |
| `bench_sm120_layernorm` | `True` | `1274232` | `cfe00d47c86122d64d9df31bb422ad946aefb2815cff5bfa283aafdbae783fe0` |
| `bench_sm120_runtime` | `True` | `2271120` | `0c3b2e39f47a3436c1a9519d2734f5e9a559f4a1f1a02d240a304e39604ac49f` |
| `train_gpt2cu` | `True` | `3105552` | `9a9ac080467c9796956cf7f909737ce6e11f4ae0144aaf6c7063ae89a21fdf20` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
