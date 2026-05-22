# SM120 Round Manifest

- run label: `codex_sm120_attn_fwd64_recovered_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_attn_fwd64_recovered_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_attn_fwd64_recovered_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `660`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `3c927d467bfef7c4200a0830ce0b5dce8cc1bacf31578ced25fa701d587b8d7b` |
| `test_attention` | `True` | `2656592` | `4d85f153e377b2364174ebdeb8bf79856c44be77a53392a6ae9f65636438ac20` |
| `test_layernorm` | `True` | `1278296` | `5832ce88811ec37b424d35c9c67098029325cf7ac18a03a01f5c17704a8290cf` |
| `test_bias` | `True` | `2089120` | `a554269865cdfe3e0fbcbc784be5963cbc8e8cbed2952f65da424fa6e4bbb65f` |
| `test_gelu` | `True` | `1179912` | `b9fef692eae4594dba1ced984757c84f9214ec35a19841c82091f66157d3201c` |
| `test_fused_classifier` | `True` | `1208704` | `096cfd0e4f823c8bda96cd8a624fc809cc9a9989d724d7bd994f62b9271a0151` |
| `test_encoder` | `True` | `1210168` | `f88c1c80163a4d2074d3977e8228c6eecb39ac4c34d78e8fe39454fcc262bccd` |
| `test_adamw` | `True` | `1183768` | `78c3e29e793cebbe4d7c79f84aee80e314293f8f90ed73a2909518b8455319a3` |
| `test_global_norm` | `True` | `1179464` | `75c0958d18aadf6f6405aaf2ffd8a0749689362bf247ab303693f72a0bf03107` |
| `bench_sm120_matmul` | `True` | `2410304` | `436d98767757583a0da79246af6698c6396ca497f4180e3a0964c30e1a4760a3` |
| `bench_sm120_attention` | `True` | `2624864` | `fcc5730d4ee232a5bf4a1389031781acc8abf702ece25f2adcdad85c5391e89c` |
| `bench_sm120_layernorm` | `True` | `1274232` | `18ebf97465a849599e9fcc16df6391b70161e23a0121494268f5183d0858b42d` |
| `bench_sm120_runtime` | `True` | `2271168` | `6eb952b1e2807c5e9ef4ace01cb075cf269f1b708d7b2c2d66fe2fa6031bad07` |
| `train_gpt2cu` | `True` | `3961616` | `82f972812bb4eece21a760307fa7031f3f701d5e87ea5a6cbbd9e607fd0dc8fd` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
