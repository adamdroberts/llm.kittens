# SM120 Round Manifest

- run label: `codex_sm120_promoted_global_norm_block256_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_global_norm_block256_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_global_norm_block256_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `622`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `1a66da40c482a1dd7ab0c2166a66b52bab703e56df1521581e0855f5e816909f` |
| `test_attention` | `True` | `1800528` | `9c51ccb8e87d3b4086133e98053133871af859f720ba986473a3f5062c0721b1` |
| `test_layernorm` | `True` | `1278296` | `32130e18942ccd6ce872c0f07e90e5d319a1f547ec4d169af16b9ad554b17049` |
| `test_bias` | `True` | `2089120` | `799b534e39c56df098b64c04bb03637590c61fed8f32d9399db91a085b6ce691` |
| `test_gelu` | `True` | `1179912` | `4a693465938cb38f8b2a9a187e981366cdfdc88cbb0470f097f99c77d795de63` |
| `test_fused_classifier` | `True` | `1208704` | `bde894b9636fff366ad8f524dc0fd223278854c52ebf43af09181097920cd87d` |
| `test_encoder` | `True` | `1210168` | `6841575241068f9d4991139884e9854b84a87c90920c68f4817e608676b35afc` |
| `test_adamw` | `True` | `1183768` | `990219f13d75152c47f1fdb98a9648357b81dfb1da8e490f87d4fcd9c8f338f3` |
| `test_global_norm` | `True` | `1179464` | `2f53d6cd0b91e3534642e618182b39da43526e9eb70f5f5f929f7c249ff8c4d1` |
| `bench_sm120_matmul` | `True` | `2410304` | `b7ed9a1da4edd0459f96712b3918f9bc947e2f436f80ac65efeeb7a745d55bd8` |
| `bench_sm120_attention` | `True` | `1768800` | `56093f69921bd3715a884e65a888dd2fc00c08712b01fed21245403dad59f105` |
| `bench_sm120_layernorm` | `True` | `1274232` | `d24eb61a45073e9a6c77a024bddb5a9256e52b1f0661bf24b27d84a833e0f775` |
| `bench_sm120_runtime` | `True` | `2271168` | `a0dd541490e88eb6611ebbc82cd228ee5421aa017c2dbbefb3c3563b22b37535` |
| `train_gpt2cu` | `True` | `3105552` | `0a6ab68f0099b7c602ef65a46b8d2fdf9ee10348f36740b612b8239a9699fe53` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
