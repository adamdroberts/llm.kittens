# SM120 Round Manifest

- run label: `codex_sm120_round_cublas_lmhead_forward_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_cublas_lmhead_forward_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_cublas_lmhead_forward_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `452`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2128288` | `01688e7a2374bb7b6f917bc7a3ee598072eaba9b5dc1f3b86d75fc5fde8b0ad6` |
| `test_attention` | `True` | `1760032` | `e5742668fce4546f6e09d4fc265b5af660173848bd53653cadeb92e8624eb9fa` |
| `test_layernorm` | `True` | `1237784` | `6f42857c672b3f534e009267d18168b35f342eb54e23d58de371081640a76865` |
| `test_bias` | `True` | `2039504` | `38bcc66cd03be8bc7c60fe5da2f64f9296405d68287b5a9e23dd7fdbf294955a` |
| `test_gelu` | `True` | `1139336` | `f3029d2d0d0a1aca0cb0aa5dd3fd6201d04cb0be090008a1154a4602c13e6a81` |
| `test_fused_classifier` | `True` | `1146768` | `3db5f5c36901533d0ffcad31b56aae3b9ccb5c8b5f42d4372f34b46f3fd773b9` |
| `test_encoder` | `True` | `1165512` | `3bcd0ef44c8245335e4103676b5ea8746daa8b3a64d9d0ed65a8d1501c7c5d9d` |
| `test_adamw` | `True` | `1138408` | `645d3f3f67b7852e56dc7211726963f5e26c5a8edeb8d7c269bd3248fd2d5a1b` |
| `test_global_norm` | `True` | `1138816` | `0939711fc44705c64d725ae8625415a819cd9010954f33e0b2f6898f0410c4d4` |
| `bench_sm120_matmul` | `True` | `2365064` | `4f7dbbd00fa2d19e9166c75d66bb0aa45a8aa43eaa1d1ed9acc3eed209c83fd5` |
| `bench_sm120_attention` | `True` | `1731352` | `d0d425c8cf30de4bde7670719c5da3476f9439976854702127f174aa03455e89` |
| `bench_sm120_layernorm` | `True` | `1229088` | `4682372479fdc052f7534d1f97cb6f2fde0910dace233f6a5cf85a421f4c9e0d` |
| `bench_sm120_runtime` | `True` | `2168552` | `ef9c4119733fe7a80942206eb635d056eb75d39a211a10f9377de88527efe8b1` |
| `train_gpt2cu` | `True` | `3037176` | `855af1d401a991ac252c3f1fbd6ea5a23b566ff94fcf9b59e3b10075e7c27b78` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
