# SM120 Round Manifest

- run label: `codex_sm120_precompute_grad_scale_adamw_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_precompute_grad_scale_adamw_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_precompute_grad_scale_adamw_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `547`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `d0245bbafd2c49f07bdf24abab74b84522248aa5f64236ddb7a7226c60c5e846` |
| `test_attention` | `True` | `1760032` | `62cdf6b7b5485dd2f3e492e07ea6f8445f113ebf3ee6f48e59d28730f753bcfb` |
| `test_layernorm` | `True` | `1237784` | `6422ee8da01e5699d23c76d1984acd587f2dfffcc7e5fab55490b5b24c692292` |
| `test_bias` | `True` | `2048616` | `056fa79c103ead736eae276a016a14f50b3d7cea339d715cd080b532639f626c` |
| `test_gelu` | `True` | `1139336` | `f3906efe5439e17a1c0f1b458fd066a23ae6a3b141317223fb5bfabc140adffe` |
| `test_fused_classifier` | `True` | `1164032` | `23b4a87fa65ad709c327e52a294a88a37df401317cad8b75bb35e9e98dd1c71d` |
| `test_encoder` | `True` | `1165512` | `257c66d0598a1ab453ceb6d83286651c88292fe30044e6b934c338bbaa1bcaeb` |
| `test_adamw` | `True` | `1143192` | `4d492db74e822d84483008a37304a1fd6ca09289145176f36c8e09957d6ff828` |
| `test_global_norm` | `True` | `1138880` | `9c42f84c358eb3d53b3537ed1ca73c7641db5cecaa0b3f86f8fdf00f6a55b7b9` |
| `bench_sm120_matmul` | `True` | `2373912` | `b64985f8ee91c9302795e49ae104c8e0eb2b140a7e21c6119eda5482d9041444` |
| `bench_sm120_attention` | `True` | `1728312` | `7917a5d74714d6861393b73194a014d8b4e220149ffcaa824d7de5b2bc9e7ca3` |
| `bench_sm120_layernorm` | `True` | `1233728` | `90bfd0d832e401ef40de05aec53af7d6ec03f1daf92c7724fbbd84a9d0b6431e` |
| `bench_sm120_runtime` | `True` | `2226576` | `fb353b7da90d23f1f3511caf7a05fad3331f2485f74cd3e7470da12e12bde3e2` |
| `train_gpt2cu` | `True` | `3074512` | `19478a5efa1517bfcb3547da69ff9545ad65728d4130a12371efdfde99058ed4` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
