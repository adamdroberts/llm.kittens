# SM120 Round Manifest

- run label: `codex_sm120_round_cublas_dinp_attproj_only_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_only_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_cublas_dinp_attproj_only_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `509`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `72cc0a2cc13d54fb423b4cf56f7e4fd27c77abfc4fd3d0edab842b51a7aebeec` |
| `test_attention` | `True` | `1760032` | `969e00dcd5c252a2b21505f547543e6b06927fe3624a63ed3d828abcab46ec58` |
| `test_layernorm` | `True` | `1237784` | `f3df7ead73a59efde338462fc02745119e756a9f3318a14693b3351ecd1f672d` |
| `test_bias` | `True` | `2048616` | `d63d8dc3962c16aa5923e5607484647f1f32d732ce2ec9bcca0ab0fcb8806108` |
| `test_gelu` | `True` | `1139336` | `10917954974664203eaddcc1272107f1f1c0880f16353088012144fdf674dd09` |
| `test_fused_classifier` | `True` | `1164032` | `96a8c0980e128ac2aaa28ad8343d4dcda804d6b9d5265b4d4d44289146db39ed` |
| `test_encoder` | `True` | `1165512` | `9759d484176e41d4d86759131670a8a5e97c820e558357cc48f397660a5d3daa` |
| `test_adamw` | `True` | `1138408` | `1b15fae740407bedd325e4631ffe690bb4118e3cd41ec281cea0f8a84cb801b0` |
| `test_global_norm` | `True` | `1138880` | `04e3eb6901ea6cccb1858495196f1b14254e2d347f39292a9d716017c5d202d3` |
| `bench_sm120_matmul` | `True` | `2373912` | `23a28863995cd60eae584510d5561d307bbb709254926750507ae1fc918721e6` |
| `bench_sm120_attention` | `True` | `1728312` | `66f73b95008750dee70cefeadf16ab54cace0ad1f2488ca760d1cbf183384990` |
| `bench_sm120_layernorm` | `True` | `1233728` | `9843ac6d987139443d968191753b38190528b677c0da940d957226d8e270e9b2` |
| `bench_sm120_runtime` | `True` | `2221864` | `1b065d6c582477ed113e08733380ad530e9528001327738edcde26f58cd64447` |
| `train_gpt2cu` | `True` | `3060032` | `53c6c7f02f5bdbe85686d11db48ab140d44da8d7ee62ef48047846a5b47f1bf8` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
