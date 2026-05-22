# SM120 Round Manifest

- run label: `codex_sm120_promoted_cuda_dresidual_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_cuda_dresidual_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_cuda_dresidual_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `608`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `b0d3ecd3da68f2b9f7dde4008a07a70968026846349e0f08e539ca60f8fa3a1a` |
| `test_attention` | `True` | `1764128` | `36cc6cdecb72c8eb0f5126cedea853ac9830a72ff13915f0573ee1e4f5c55932` |
| `test_layernorm` | `True` | `1237784` | `6eaed380cbee46477c02decea4678179480ef1dd258532d538bbb8f41b946656` |
| `test_bias` | `True` | `2048616` | `f4c018d038e7acfa4a2a1e229595aea71cb1a748bd92138f1b5b386719f28d8b` |
| `test_gelu` | `True` | `1139336` | `006ac995695329167ca836956c18b0f01145b23ad7afa7426d6b6b32e6774cd9` |
| `test_fused_classifier` | `True` | `1164032` | `a24de6729574d91093ed24492b381291d40b88a7934d7d8e551cd87650df3bad` |
| `test_encoder` | `True` | `1165512` | `9c5a5449aeb71a2763011de851dc13cc4624086e7c61bed99fba8826c7f9a7a9` |
| `test_adamw` | `True` | `1143192` | `94d6ff0ff35141f9575e08501121823bf6cbe6983f076820360b9bcc67b10d90` |
| `test_global_norm` | `True` | `1138880` | `7dbabe1885af05df4eedc62058f2c02c008211ebb419321093ae77977b29fd28` |
| `bench_sm120_matmul` | `True` | `2373912` | `62329e7387d569cd1d55a883102fc4d050a63c7d7d522075fcfb52593cd3d9d9` |
| `bench_sm120_attention` | `True` | `1728312` | `6471751ca386d284562a794ea7e6159258efd02291cd635ffaf7fe1e402003e6` |
| `bench_sm120_layernorm` | `True` | `1233728` | `655376c1956471fc1bb8d0fddef5499e8ae836d98ebf73ee08c99fcfb48d64bc` |
| `bench_sm120_runtime` | `True` | `2226576` | `a50686f0c67741dfbb60f78e03697845bd7790ad027a2c86937864e331580a26` |
| `train_gpt2cu` | `True` | `3064992` | `142552ea94123b06da93d651756367f20d36507b9e7f07e94d8acb2ea53e2b43` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
