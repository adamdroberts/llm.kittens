# SM120 Round Manifest

- run label: `codex_sm120_fcproj_precompute_grad_scale_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_fcproj_precompute_grad_scale_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_fcproj_precompute_grad_scale_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `666`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `3863e61406f15390adf8299c86c1a8ed7e02b3fc7a80a640717a503beb406e49` |
| `test_attention` | `True` | `1800528` | `3fe2b02eaba7f393c646b606d0551078fe7f1de68423dfcd3a891b9ef9e15fb6` |
| `test_layernorm` | `True` | `1278296` | `629f6163d1f8f814ca9839755926ed39037b41a3b31cc4ca227cc5674ef7903e` |
| `test_bias` | `True` | `2089120` | `b5c8362a9e8dda8efb70ff5c6208a0354a34b1e2795aa07d0b4501d46484a79a` |
| `test_gelu` | `True` | `1179912` | `50cce8b5fb5dfd689dcc6f589cf282c477584bf50703c39823dd8c192734a84c` |
| `test_fused_classifier` | `True` | `1208704` | `ec571c23b44c8a0a4748527f8e53e16434db741a9749be5c38fad9d2b9273b5c` |
| `test_encoder` | `True` | `1210168` | `afe487f6d4d72d004731b4905d81feb0bfb5413185d0e2be7e7d8fc7f57a0d5f` |
| `test_adamw` | `True` | `1183768` | `25cd0aed0f5ee13c203c084625b3d54f102546bf9ba1335a20fa6125218a5e6d` |
| `test_global_norm` | `True` | `1179464` | `1e042b57675b778d3dcd4e0b3107429a1bc6b11344d1b36e41db0cd157f1cae3` |
| `bench_sm120_matmul` | `True` | `2410304` | `7a7c16f058fe4bdde7753340c44eb82440b3815b025278f05ea66bb2e6e209aa` |
| `bench_sm120_attention` | `True` | `1768800` | `5bcc7621f9cf747c2f0f0fa633c35d31a91bdd9d6e759183ef7edf71c2124375` |
| `bench_sm120_layernorm` | `True` | `1274232` | `96928682e20e8da9803b84920425c2225b56ce5c041efaf0cee3f879bf366541` |
| `bench_sm120_runtime` | `True` | `2271168` | `c4628ef1281d6194d9e9121c6cfdc2176a3ae7d722278b61ec830f03d9d30c3b` |
| `train_gpt2cu` | `True` | `3119216` | `b8b27b761a5da7788a3f6b5d28d290aae1027c72d50963f1930e60c0940b9186` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
