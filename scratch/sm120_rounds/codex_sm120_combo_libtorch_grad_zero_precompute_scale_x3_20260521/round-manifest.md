# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_grad_zero_precompute_scale_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_precompute_scale_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_precompute_scale_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `552`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `93641f1911fcdf321ca6c8fbdc14e3876f48b44102484f9712df31f863817ff3` |
| `test_attention` | `True` | `1800528` | `fcc61a42d76b6807e2a98178f252661d013b6b2c1b4b8474174297db95331bd4` |
| `test_layernorm` | `True` | `1278296` | `10b5184c18e608b3bed7569746e1a7ef585519b2130203e8595fdd12f2051283` |
| `test_bias` | `True` | `2089120` | `a514ee28adfa165d5ea76565f86605adf746223cd5ce00f85f1b14b69242078b` |
| `test_gelu` | `True` | `1179912` | `5e0e87d3de84b4ec0307d8207cb3fef1d123c747808bb718d891caf7ef87b828` |
| `test_fused_classifier` | `True` | `1208704` | `ac09cae4e36ac79c119ea4206c29c325e1652fd7eb6d0d6abd427369fcb759de` |
| `test_encoder` | `True` | `1210168` | `1792275aae4d504915f2cc95bf8a9879133ab9059f7e684eaa6fa8175aaf96b7` |
| `test_adamw` | `True` | `1183768` | `3415b74364386b35406f866f60b71c0edf5c8641917a8acedd0e6a93f7626b7d` |
| `test_global_norm` | `True` | `1179464` | `1aa7b7b8452ef51c0bff9f5e8379c235eaf355b09208493de43e3e5054c4f73e` |
| `bench_sm120_matmul` | `True` | `2410304` | `e96132abf7855718119875a525f69d9398fd21c0314d15c2e410c234d7d98b65` |
| `bench_sm120_attention` | `True` | `1768800` | `2babe08e01489507a48da6b5d1dfcdfcabfd38f542a444949ea94a8d7b6cdee7` |
| `bench_sm120_layernorm` | `True` | `1274232` | `64457cc10a5e1781687206d0183e74c31df5856739ea9d077c2b5249aa6f7761` |
| `bench_sm120_runtime` | `True` | `2271168` | `ef3c03e808d704f95ad940269aba0ef0402390d97204bef8cc30511aeac1cdd8` |
| `train_gpt2cu` | `True` | `3119216` | `c5dd0e2b646b047a6844fe0e45c1fd676418d9e2e9b9e2daca6dd66c683fc03c` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
