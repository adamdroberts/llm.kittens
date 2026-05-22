# SM120 Round Manifest

- run label: `codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522`
- device arch: `SM120`
- max steps: `10`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `686`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `06b3fdb4cffb6bf4d85779b508a5c409adfc4f8f037843261f5152d4cc1ca164` |
| `test_attention` | `True` | `1800528` | `4ddd1b18a206ee236a2617d46e63de5270496326d51ed980182ceacce6ab08e1` |
| `test_layernorm` | `True` | `1278296` | `b509dde21bddca54d62b39e2393cf070ddd4a948713e899b479e4412e928eb8f` |
| `test_bias` | `True` | `2089120` | `b1806ecfdca105cf99b7c21ef1939303ae3828b3b0a8f7fa202a446c64a08b2c` |
| `test_gelu` | `True` | `1179912` | `c7359435b21752da2ac64b12bf436229ae6df77b725d793a35e19a581a4e10d4` |
| `test_fused_classifier` | `True` | `1208704` | `a0d8230481c48b8bc1c888d4a54164b7e6b9f050b979cf3d159ac1d47acdaf1e` |
| `test_encoder` | `True` | `1210168` | `f9264f861c64c4f07f73c980fd9b2ce3b79bf684a22d3b34fd511effc172dd8f` |
| `test_adamw` | `True` | `1183768` | `876e6278a0e2503f27e3fc2a74876ab07ff92c8ebc9259f3597a7f973bb4896c` |
| `test_global_norm` | `True` | `1179464` | `6d1e93a4bc55dd1f17f5de2ace0ee9893d7a9449620448bc5cfb2dc93dc7b398` |
| `bench_sm120_matmul` | `True` | `2410304` | `6dfb18a71d8c971b2d7a6093526f528f63c398b53da83073d9782400c43a46a9` |
| `bench_sm120_attention` | `True` | `1768800` | `3deb3ad16f96e57ec8addc4aa2951fde8e58f64729d56ac932a18faea2e70ccd` |
| `bench_sm120_layernorm` | `True` | `1274232` | `29277cb6dbbbe7c487b4a58761b94d74c1e3f6bded74202b85e01be8e482c381` |
| `bench_sm120_runtime` | `True` | `2271168` | `d24025c315a433cd6a14ff7a55cbea950da82241751bdc263b7fe844b812dbf1` |
| `train_gpt2cu` | `True` | `3105552` | `7a1d62cb53f7695016989eb1498497b129b35d156d590ada9d2ad4770991238b` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
