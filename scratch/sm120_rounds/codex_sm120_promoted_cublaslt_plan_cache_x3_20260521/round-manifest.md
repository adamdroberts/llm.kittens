# SM120 Round Manifest

- run label: `codex_sm120_promoted_cublaslt_plan_cache_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_cublaslt_plan_cache_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_cublaslt_plan_cache_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `623`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173704` | `a44ebcd7e5c088699d9c9955105bfb35a296c706162f33f8bf0212a2af6bcead` |
| `test_attention` | `True` | `1800528` | `6132320f253c26879633bcce73497e55c134756c6350cda513f779df0dac679c` |
| `test_layernorm` | `True` | `1278296` | `1502cb3d7898ad6277ca0749b8db16e306a1d4b605cb84e806dd0416e90f6545` |
| `test_bias` | `True` | `2089120` | `cfd7d3aa328d97792f6b10c9c8a1aa4ad31e5e68287671f61a9ff9f027264d10` |
| `test_gelu` | `True` | `1179912` | `9d5c6a512818285b8c45e8014bb5e55cf783f949353a6adc7e1f8168ec5df384` |
| `test_fused_classifier` | `True` | `1208704` | `3b9b4291f4542b5b7e55332dd93d9421ef023de40086522d8f2addf20e46f3b1` |
| `test_encoder` | `True` | `1210168` | `4dfcd41dd832b0680c2be6d69813ae44df8bc61f4962f48ad62ab40ee52c21ff` |
| `test_adamw` | `True` | `1183768` | `ba6a5ecc69ea82f9ccf5d59bb3142e5eb8f29bb8b0bab639ca892fc7b1f26e73` |
| `test_global_norm` | `True` | `1179464` | `08c0115e27a99877306ab712534a25410e47351328c0a6ad5ee1a72e01b8e5cb` |
| `bench_sm120_matmul` | `True` | `2414272` | `25a9360471b8f32dd91a6e6baf5c6af035dc9d1f76078bc9d60c1a6301798af8` |
| `bench_sm120_attention` | `True` | `1768800` | `1e277d7fcfee5ceaaaaa9106ee311f55ccddbaa0e51e637c9f69e497d01aee31` |
| `bench_sm120_layernorm` | `True` | `1274232` | `c5127d39988f9be85498104bec5fe28da6fdc13777382cca5c2843481bb0c857` |
| `bench_sm120_runtime` | `True` | `2271168` | `eefc002c64238a207ce1647426c09e5c9c779e58bcd92ce89be69d43f6e616a3` |
| `train_gpt2cu` | `True` | `3109704` | `67655be0c9df246fc5d0dde0157ed77c4895f7228c8111ccab6146c7086ca689` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
