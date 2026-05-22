# SM120 Round Manifest

- run label: `codex_sm120_round_adamw_nomaster_specialized_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_adamw_nomaster_specialized_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_adamw_nomaster_specialized_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `466`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2137216` | `05233ad98b9018c5b481394ab0bca98ded5468bd801ae76bc09d9c73fa94295a` |
| `test_attention` | `True` | `1760032` | `8f4b6327b074c479f0e4ada64a30fcff6282ffdbca226ec34a828c7a0dcfcae6` |
| `test_layernorm` | `True` | `1237784` | `eb4164ff221fcf8470d05bda8cac5ea2cba05c1d050f2de6b2d037d24236fd7b` |
| `test_bias` | `True` | `2048616` | `faa3641e0c5a2acb2f48634385cb20871d621a6e59fced3dd6cc07fc2285275c` |
| `test_gelu` | `True` | `1139336` | `53d0aae162e16f78267ba80b2a5afc4697e1a3d725c4faaa8115d2620d9a9d38` |
| `test_fused_classifier` | `True` | `1164032` | `8227e1f47e835a6ea6773aed660ce7fdd9e80a871d55cd2933ccfce935a43059` |
| `test_encoder` | `True` | `1165512` | `d5032c4e4a716e5958a9129f573e243b8c9bc08a608e3eb789275e251c788979` |
| `test_adamw` | `True` | `1143208` | `19ab9150d851656b5136d30c2426e3676b16f56f5e9ffa508cbb895453e23970` |
| `test_global_norm` | `True` | `1138880` | `cd64f7610149e2ed91242fdcbfd27857201252c9d793a9fcd17f3d19c601081a` |
| `bench_sm120_matmul` | `True` | `2373912` | `e5e5ca3ffd6d06d750fe81479378539a315094e2a3ba93de395fab2704f18429` |
| `bench_sm120_attention` | `True` | `1731352` | `585ec08c2590a304dd85d4cd36bc9d40a7a06e9eaadc47ca4996b1e4e3d91461` |
| `bench_sm120_layernorm` | `True` | `1233728` | `c388159b8dc0777159ab20edfbc21c4900dce505e002a248ac3f50cc7221689d` |
| `bench_sm120_runtime` | `True` | `2207936` | `5c201a6f4815c9ffbb0fce0bca7acda426789dd826219a7a0e6b0320b2454da6` |
| `train_gpt2cu` | `True` | `3054888` | `d116e1f4ebc642affec93931817f5cadbbdfd63901cd9f7a07b6de3de65225c3` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
