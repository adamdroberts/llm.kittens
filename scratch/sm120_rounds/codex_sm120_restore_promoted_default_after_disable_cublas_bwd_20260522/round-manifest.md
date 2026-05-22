# SM120 Round Manifest

- run label: `codex_sm120_restore_promoted_default_after_disable_cublas_bwd_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_disable_cublas_bwd_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_promoted_default_after_disable_cublas_bwd_20260522`
- device arch: `SM120`
- max steps: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `643`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `2a5a81a7dd01fd494e9e84ffb5140959c6861641793496fe52295606fa986a6e` |
| `test_attention` | `True` | `1800528` | `809f9cf2ce22e44efb512d9da3b802f5009029a82730cb5dac9601e5ff2a576b` |
| `test_layernorm` | `True` | `1278296` | `92d3154bdd455cdcceab43b5fd7f52ed629d2420a7c1d2186beaa8d9b4b9d4e7` |
| `test_bias` | `True` | `2089120` | `df4d45f12d450021f0caa86d4a02b46a8749a32305233e7877e50a7400729344` |
| `test_gelu` | `True` | `1179912` | `ed43f0aaf4617b84b257dd36c885bc833d8c2bb425ff535642fb5654820c169d` |
| `test_fused_classifier` | `True` | `1208704` | `6f604c02ba94fa7e698de28d9c51877de510b6fe2f2118d916193f7b53c61585` |
| `test_encoder` | `True` | `1210168` | `52f7b90fb77fad6bcac8b3b0631f04919b2c73fd490ee703b69904326ea73209` |
| `test_adamw` | `True` | `1183768` | `f26b411f64e04cd7a5f954de35a4721748a63b4ca6b07aa5e814ac4d0d9c81ef` |
| `test_global_norm` | `True` | `1179464` | `0ae7c0ee7419fde3c6873bbb15c676e0a2e14a51e438e6f8713037c5fce74079` |
| `bench_sm120_matmul` | `True` | `2410304` | `a3e4ce947551b938a3ebb638776b4883dee1e27f7ef16b8abbfe7984c5b9ccde` |
| `bench_sm120_attention` | `True` | `1768800` | `132696c346684ddbe24472a9842e2ebf665eebffd34c26c3ab309e93c4607fec` |
| `bench_sm120_layernorm` | `True` | `1274232` | `d3a37898e7af923f469f74544467855fe7157291e8f34c3d14ab82be23cdee97` |
| `bench_sm120_runtime` | `True` | `2271168` | `b80db234e5629fbfe2fd7637ebcc070cfeae3eb7708704fdb13b1b7096e213c4` |
| `train_gpt2cu` | `True` | `3105552` | `da47e81e5a54dfa3ae1b1cbe3d5fc052ea94b84d9ad4216e466ff4c420edc0ff` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
