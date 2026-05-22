# SM120 Round Manifest

- run label: `codex_sm120_round_current_native_x10_median_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_current_native_x10_median_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_current_native_x10_median_20260521`
- device arch: `SM120`
- max steps: `10`
- git commit: `0f21747`
- changed paths: `487`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `fdc25e650ee6bd388328faa4f689efcbe38dbdc3330a903239f246a09a2a6199` |
| `test_attention` | `True` | `1760032` | `811e0d18ae0b2e2d0ba88e19d113b5c34fe4ef936d97df2cfc1f228496b516be` |
| `test_layernorm` | `True` | `1237784` | `a1694ea78ca8cfad74daf9ae2b2ab7f1e89f9825dd42ca8993698bed5799ec1f` |
| `test_bias` | `True` | `2048616` | `ce2e4feae241ae8da84fc19d703a20b07c68fe07355cb2cbffde2f906c5ac07d` |
| `test_gelu` | `True` | `1139336` | `612ad87fd1e4a8a7c6756270890cb0281c57b5137222ee80a366e393448e9099` |
| `test_fused_classifier` | `True` | `1164032` | `3cfa3d102bf5567640f6a8a42e829dcc6f6da1cb04abc08ba1faec0460f135af` |
| `test_encoder` | `True` | `1165512` | `41bb581688d61ffb8b59b6149a00d34d6096708e94c160d14ea32fdfc2694ea8` |
| `test_adamw` | `True` | `1138408` | `a3229e99b73faa51c9e8dd064e51deed9914e15df0de72b5a0ace308b841a0fe` |
| `test_global_norm` | `True` | `1138880` | `300ead1e66ae31de504c6c8c17f71c717cdac6fc1331d0d2977eaf78e1e4de18` |
| `bench_sm120_matmul` | `True` | `2373912` | `27950af54a948cef38175aaecc05a7fbbeafaadf89e143e3bcbe7f3d15c02d87` |
| `bench_sm120_attention` | `True` | `1728312` | `24f4f18f87e5466f2ece6a1206e24106496f58ac88889c86ee384013add17e6f` |
| `bench_sm120_layernorm` | `True` | `1233728` | `22696f40d0798f04fa0f32b142410af666ae40cd3b5b06883835fcd4ae07ad24` |
| `bench_sm120_runtime` | `True` | `2217576` | `1e4ac34ff02a690f0029a220cc15a04ae06b1d72d8c7655a1ee4b4042659b92e` |
| `train_gpt2cu` | `True` | `3045944` | `82a43181c89ae6e29bef16516bd1da9268d7e0b0db819669725c4f4de46733db` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
