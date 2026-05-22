# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `575`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2190792` | `88e358e1f4a4cdbc9cd3085bbacaa958b4fa7e82fcf08634598fc9c24ce94e44` |
| `test_attention` | `True` | `1800528` | `5837b06ce1da69c5f521d6058687628a9f94761478a4e288d6d7f35aad3f7bfa` |
| `test_layernorm` | `True` | `1278296` | `a8fed12646d8ff0fa9d0b9c76c4baea4947088854a1c9aaa12f80826bbf56096` |
| `test_bias` | `True` | `2052256` | `75ba8d0b2e287088e77a04d9f8db389fe9fd1b12d3de53623551b711601d7dde` |
| `test_gelu` | `True` | `1179912` | `2fe92caeb68043a4affd2057605a23f0f4da35a797b941a125c1de519834258b` |
| `test_fused_classifier` | `True` | `1208704` | `29cd7467bb97dcfa210851bda9ce105836447edeb5ed3b9c39d7dc5adbd578ba` |
| `test_encoder` | `True` | `1210168` | `c9e2bb25bb6036eaefa67bb6171ebadea98f30c7258373a4e88296fbd95416f1` |
| `test_adamw` | `True` | `1183768` | `24cf5c2a8e38d0b71fddb1c41b2c4f5483c29a7c2fc1b1f3e2fde9e8a1427007` |
| `test_global_norm` | `True` | `1179464` | `1e1490ecd65438d30168fcb1c3d5e15837c43ef5d011eb3eef0ec7bf26d1e87a` |
| `bench_sm120_matmul` | `True` | `2369344` | `c5f30e5a9ee34c3b28dbb36699d7900d99f89abea9cc8d07850e1a10202a7650` |
| `bench_sm120_attention` | `True` | `1768800` | `99c55597912a2df7996be0c5b952bc90636341ec27394251926b4f21e8fbd7f7` |
| `bench_sm120_layernorm` | `True` | `1274232` | `ab9dde9bc98617cefabc6d10c9683155142248d33803d95f45afc5027a905c3a` |
| `bench_sm120_runtime` | `True` | `2234304` | `54baf60302e78239383ccffd71e191626498c064d6b7073c2b8e48380b79fc14` |
| `train_gpt2cu` | `True` | `3107416` | `ade6de9b4c1a1c5e8143de426313e9fe256d8eb29734101ed600facf16ed3b09` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
