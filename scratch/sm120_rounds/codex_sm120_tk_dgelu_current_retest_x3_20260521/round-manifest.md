# SM120 Round Manifest

- run label: `codex_sm120_tk_dgelu_current_retest_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_tk_dgelu_current_retest_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_tk_dgelu_current_retest_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `522`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2154392` | `9d9a0ceee1cd781a3034bb67aeb4d25ad24178826a648250365eb6e08cdeaaa6` |
| `test_attention` | `True` | `1760032` | `6589632c19537e4bc0b85f4c196da31d968da65a2acc4946cfcb2fead87ad692` |
| `test_layernorm` | `True` | `1237784` | `760c910f5ed62c739f703b8ba1b7e903c6f0d688724b3866f8a10a4bc249c920` |
| `test_bias` | `True` | `2007656` | `f2c83038c8d7dffedc7563378dec9c74bd7e4998c71f8d24fd605233fae21e23` |
| `test_gelu` | `True` | `1139336` | `a70b8328fd6d22701e68b68b74e77570bf18b331a0245d312b88eda73d96fc1a` |
| `test_fused_classifier` | `True` | `1164032` | `85396168c2bc0c7f9d702130e33e93bd9e0e73949dc966c71ab5f707e213ae16` |
| `test_encoder` | `True` | `1165512` | `ab3bfd2b0dffe4b77869dbf0c4beef28f9862e299f026f682ce9b90e9a69a526` |
| `test_adamw` | `True` | `1138408` | `52d1880fe0f9a4789f9bf5bc7ce5504d2a6842835fdf1d97572769cd7e7aa015` |
| `test_global_norm` | `True` | `1138880` | `abee92794a875b6085a7a281120f13839d5e111077cd571af38c4e98dd2f72d8` |
| `bench_sm120_matmul` | `True` | `2332952` | `ed86d72cb7ad687287bbdd5e3bf586a34afcc392a281cac5b73483a2fb0b17a5` |
| `bench_sm120_attention` | `True` | `1728312` | `68c7d40cc70719af6eae7484d96d7badcc16f38bfee613da50d6443268bb8775` |
| `bench_sm120_layernorm` | `True` | `1233728` | `18b2d95861f31e2f49e5b278cb1008a08b2857bfceaa2a18fe65b175d2368615` |
| `bench_sm120_runtime` | `True` | `2185000` | `ff53594c5698b85708f2ae9e77c59cd3bcbbfe02a197cae0a4a5ca20f019f676` |
| `train_gpt2cu` | `True` | `3054992` | `daeb79526c2d04ce27959eb3bc6079b89b9357ea826bb06d230a858d2df1023f` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
