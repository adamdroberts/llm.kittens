# SM120 Round Manifest

- run label: `codex_sm120_combo_bias_wide1024_layernorm_bwd1_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_bias_wide1024_layernorm_bwd1_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_bias_wide1024_layernorm_bwd1_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `558`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `22c988bce2412fc79c892671048cb524f75bfad6b8510cf37020cd232ed763f4` |
| `test_attention` | `True` | `1760032` | `48e253906f06b6fae597b655c29b305f8d5a459a3dbf1e53b04099fc81fe47a6` |
| `test_layernorm` | `True` | `1237784` | `5d71d06d2ba2285d50981ba3f59f6e1175f0845863c024c773c14038e41a5610` |
| `test_bias` | `True` | `2048616` | `e397b4d45c8a9704d6ede0dd1a26257f39259dada0405c9777215174f75e7fe3` |
| `test_gelu` | `True` | `1139336` | `1fd94311fa6464db0d01abad755943fd415c533b87b6d404063a6a965af1056c` |
| `test_fused_classifier` | `True` | `1164032` | `76112319c742c44b93505e117afaf1604fcea43d90688c0478680d09c7195f96` |
| `test_encoder` | `True` | `1165512` | `048b5c9b98e1112c3a3b36c5c27a9b1c7831fbd55aac45949b979b2e05973239` |
| `test_adamw` | `True` | `1143192` | `ddf9094cbf729799fb0c78131b1cdc624d1520f1dda210e8de83d70a7faca4ef` |
| `test_global_norm` | `True` | `1138880` | `4bc78b75d2cc4c9e3da430bc0faeaf8254bfa2bc1794b3debacec6edb8b6f137` |
| `bench_sm120_matmul` | `True` | `2373912` | `b45bbd3cbfa570bc42fe7c0f6adaebed5dc68cf63196f0973af8f698d3e04ce5` |
| `bench_sm120_attention` | `True` | `1728312` | `dbc0b12ca7d3841657a4b188c79a3c0efcba63f29fcb4fe10f84611abe602206` |
| `bench_sm120_layernorm` | `True` | `1233728` | `ddf865c2b476699c96afb522fe3d2e3baac952d06235add9fe5312be67d43f60` |
| `bench_sm120_runtime` | `True` | `2230632` | `d59a4442f9d6dae11bb67ed93a96f91b1829b6e3a3489bb5d135d8f3fb46428b` |
| `train_gpt2cu` | `True` | `3064992` | `3da1db0bd7a4f0909c8781eac2c8f125fa41c1dbe961cf19a60d010ff63ba6f3` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
