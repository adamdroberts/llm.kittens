# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_grad_zero_all_native_winners_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_all_native_winners_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_all_native_winners_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `574`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2190792` | `6f0163c5e3fc9aac104f289ca863ec49c74e7920eb2927e5df475b2612b6ab92` |
| `test_attention` | `True` | `1800528` | `d6e1962745735429c295fe0b62de80626bb1d4d72fe57b26c2c1748096addaeb` |
| `test_layernorm` | `True` | `1278296` | `1b9dd9cdc8154354aac56dae5408c306a0ef51ba94e04cf6e9db38d44b58efcd` |
| `test_bias` | `True` | `2052256` | `b0c27b71644477eb95a82607251ab26a73d808cbebfd48ced65e28ad198a565a` |
| `test_gelu` | `True` | `1179912` | `85169a03e8b33feb6d4c11b56d2e6a49785a116a946fcc8ace434d66d6e9e1bc` |
| `test_fused_classifier` | `True` | `1208704` | `25a4cc769854bb1b2ae2378a605a90d05febe9113c569c9424cdbeb790964728` |
| `test_encoder` | `True` | `1210168` | `8b8530e97d913c7d6d3b9abe8747b90cf16a16d9142d3c21cc48b9ae5a3b5bbc` |
| `test_adamw` | `True` | `1183768` | `b046f95b8d47ccceb84f45e47bd76d654888e4bfd5e368f211a5ac0e32df0cf0` |
| `test_global_norm` | `True` | `1179464` | `2400f11017d92f93c2bac21d81fc6e50c666c65558488ec47fdaa9b367b7e8e9` |
| `bench_sm120_matmul` | `True` | `2369344` | `96cfa77207c285e4e63c15783f4c4abe3079f69320fad6f41703581f56377fd9` |
| `bench_sm120_attention` | `True` | `1768800` | `569fac08bdff89fee23a4d48ea52ef21ec3ab1f6306b41378d4e2b98ab30fd58` |
| `bench_sm120_layernorm` | `True` | `1274232` | `0b67869361d26309d7f9e9ddea7fbef317ccc04c5d1a9e1f6e23f44ceefa1a32` |
| `bench_sm120_runtime` | `True` | `2234256` | `b29d3187dbde6e44cd56ddb5dec91be5a660f7d472ecbeabc4a65461b0e4365a` |
| `train_gpt2cu` | `True` | `3107416` | `82147a41a44ab94f9c6a34e5ea0517c96eb51ec121f3aae720642a70d1e0f113` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
