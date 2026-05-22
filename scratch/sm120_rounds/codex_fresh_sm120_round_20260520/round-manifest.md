# SM120 Round Manifest

- run label: `codex_fresh_sm120_round_20260520`
- artifact dir: `scratch/sm120_rounds/codex_fresh_sm120_round_20260520`
- train output dir: `log124M/5090_S_codex_fresh_sm120_round_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `430`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2115600` | `193282ed67f338f1e2e49d8ed0aaf949a6e2f674e51ba45cf0ba5efe53a31db6` |
| `test_attention` | `True` | `1760032` | `48e69f1275d36e6518cf9a86a8b04566ff099e9e10e8042dab3b5ceb934196d5` |
| `test_layernorm` | `True` | `1241880` | `4751e157cc8e2e47d544371c6ec43a4f2229c1dd8c13908e9730610bfbd72d54` |
| `test_bias` | `True` | `2039504` | `28e1b4fc493047918ff37d5a344c594884ad11d7c39f668e008b32ab0b943c55` |
| `test_gelu` | `True` | `1139336` | `a12d364ac0a6329d7b836892df83a3b83c0c0f8a3fc4b0a6f6c793b9a51226bf` |
| `test_fused_classifier` | `True` | `1146768` | `a4feab9b52dcdeb2d294d8c29e3e9cc0f072184de57e0c29cb87a4967528994e` |
| `test_encoder` | `True` | `1165512` | `5a10598ecb6770932e073c68e474e120f31b2200b33070a5c725f9987e130440` |
| `test_adamw` | `True` | `1134168` | `5b4cfcf4222e2e4e1c11049ab79e20a9b04c7f4bad980429e582724ae4e78cc9` |
| `test_global_norm` | `True` | `1138816` | `1260cc3d258624d79fead3754a718010b14cf1b8b1190429780922f7f6608381` |
| `bench_sm120_matmul` | `True` | `2344376` | `b5fe3c007d0d2a02e29166cc9995e7e01215b994456ebaf07b3dfdcac2781f89` |
| `bench_sm120_attention` | `True` | `1731352` | `c6e9cd4a34843a508376cd7e3105725edaf6a05590f907a28dfe79a016a31915` |
| `bench_sm120_layernorm` | `True` | `1233184` | `2c209bd377bb1c68957abc9025a087993fdeb5e80a52cb315bc0c95652daa8ef` |
| `bench_sm120_runtime` | `True` | `2168552` | `896d6b3ac31429fbc4105f97723c659d9b51adc1cb9311675d4c7f414bf8d7b3` |
| `train_gpt2cu` | `True` | `3040632` | `fe2f9224ba873092bb583aa58c9dd0b5d964e14e5f6f65968714422ebc2383a4` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
