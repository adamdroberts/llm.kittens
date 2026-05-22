# SM120 Round Manifest

- run label: `codex_sm120_promoted_libtorch_grad_zero_tk_dgelu_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_libtorch_grad_zero_tk_dgelu_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_libtorch_grad_zero_tk_dgelu_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `617`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2190792` | `501a12d141a48681d2c9054295745e08f3dbc1420450798580fa9b362736d303` |
| `test_attention` | `True` | `1800528` | `9953888ad66f6bd40697b3dc485f7c01c18bbdc5886b3bb101232d52495f3ecd` |
| `test_layernorm` | `True` | `1278296` | `1b866403a10d507e0874a3ca6630e5a011b588f686a49398a6250cd5d2d362e9` |
| `test_bias` | `True` | `2052256` | `91004d46ba14524674e7f64a19596b678d5de360fe6a3c217df7ba755c7a1667` |
| `test_gelu` | `True` | `1179912` | `443328ceab0ea607cc329fc233e4ff84f68a1a114926dc5138e558020cfb3ba1` |
| `test_fused_classifier` | `True` | `1208704` | `eb2d881ac4c5ac1fa10d6c6845d51fb6f7b020bb4429be0e030e31ec30f6edd8` |
| `test_encoder` | `True` | `1210168` | `70d7078cdf09cced27e4cf88a9de1d4e7107aeaa2411d24086f99f66994124f2` |
| `test_adamw` | `True` | `1183768` | `7af718ddd47bbeec2602f7230e8b24a48830d0a2ddb0f8c0a2704689845179b4` |
| `test_global_norm` | `True` | `1179464` | `ca1ba8320d9bb6aeca1587c13b3d50443267eaea25e1e941acaaec6afc210252` |
| `bench_sm120_matmul` | `True` | `2369344` | `bb3673ba84dea2b8959c15b3ad52288d8183efe797169a307a4aeed9f3959513` |
| `bench_sm120_attention` | `True` | `1768800` | `a1d9a6024bd9643b94885864f327b9874e5e242013c8fbeb6d867615fafc77d6` |
| `bench_sm120_layernorm` | `True` | `1274232` | `419f4a88106a66e4c6241b7622a12c8ee03b90e9383c173e984fe01d797bb64d` |
| `bench_sm120_runtime` | `True` | `2234304` | `2e12ee8a5d4467eebce317982abad42c91700cdca0233dda771689fdb791d31c` |
| `train_gpt2cu` | `True` | `3104696` | `c20620812ea1183f41a8fd7e3b1a90d678200a70277cf2235fed33fc8d6229ef` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
