# SM120 Round Manifest

- run label: `codex_sm120_runtime_grad_zero_dprep2_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_runtime_grad_zero_dprep2_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `674`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `bd25350c276306f26b2599ff2eb9c462d4deb48f265b34731be1c7f9faaa1caf` |
| `test_attention` | `True` | `1800528` | `c264a603fbf807bc25857691ff9b869df4235e31bde9aec726523e4528878669` |
| `test_layernorm` | `True` | `1278296` | `3d317ae105b99383f2a80af43dd6ebc5cfb2aadcf861f63945b3617fbf5581f0` |
| `test_bias` | `True` | `2089120` | `11408cfee970c9adf2fe23f2f33eaa1b2f3fd89250f3e8b0da5d9275551208c1` |
| `test_gelu` | `True` | `1179912` | `f4f8be99630c3b5f9848a530494bdc0819619ecb2b0bcce3078685cd943ecf54` |
| `test_fused_classifier` | `True` | `1208704` | `177fbd98afb78fe4d350e231360c5685f953531cac53e6b8f8300f7f7c5da0f3` |
| `test_encoder` | `True` | `1210168` | `ef5e6f62a57aa221c4c3fd3620b3639446734ea400cc403c3c79505f156cc8d0` |
| `test_adamw` | `True` | `1183768` | `31d392ecba80ebcf617ed9c2a795f4f6a3472a9bb40b32fea4b1c60fabe6483d` |
| `test_global_norm` | `True` | `1179464` | `8b94688feabfe627f710ed83926f6c06c01b308b0cb768f5b9737465a51ee6e7` |
| `bench_sm120_matmul` | `True` | `2410304` | `afa46022109bd7128315ff0bafc6c7c78edb375462ad6a37ac677ac8c1d2031d` |
| `bench_sm120_attention` | `True` | `1768800` | `358891753bb1e960f7c8e16138af8743707caa620666afe136b68230148c613d` |
| `bench_sm120_layernorm` | `True` | `1274232` | `823062398e3d80155d25a9f3096d4cee8a4f20645dbfcdbfeac7fce5a529c175` |
| `bench_sm120_runtime` | `True` | `2271168` | `adcf433dcfdb6b81dc853598bc3ed1aa531df1918557eee85fca2102dbdfc169` |
| `train_gpt2cu` | `True` | `3105552` | `6e62499a9314f357cb23646de4cfb5330bfc4b9776701e8b93f5a104180f93ba` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
