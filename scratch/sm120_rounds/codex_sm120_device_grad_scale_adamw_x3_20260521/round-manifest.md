# SM120 Round Manifest

- run label: `codex_sm120_device_grad_scale_adamw_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_device_grad_scale_adamw_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_device_grad_scale_adamw_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `546`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `ae2eb8a1a8b3cba4d7468db70dfe6f94e10866ac232fc3b2e18d93268e699ef8` |
| `test_attention` | `True` | `1760032` | `386285b1d94311fbafea8cd329959b639566fe7a4d8d206c24132cb4f317c66f` |
| `test_layernorm` | `True` | `1237784` | `d6ce3f8a460df2465229b1cd66e93cf439fd4547903a3f8cc0732ec7876b132e` |
| `test_bias` | `True` | `2048616` | `23c5944d4f88160501fd4a4f98da0b6f5a4fce902b33ef386c7f04d409af2e4e` |
| `test_gelu` | `True` | `1139336` | `714f637c729748eb2670832e151ca004af8853d062c159f716869eadb8cd9cea` |
| `test_fused_classifier` | `True` | `1164032` | `da0dc1fa7058717ae329b86101e75267b6786a0e389b244f0acda15866f4c459` |
| `test_encoder` | `True` | `1165512` | `6c378c6d7b7a79bb9110c7e23a109551f7a1c48af82d400fa8f70ebbf3718304` |
| `test_adamw` | `True` | `1138408` | `d1c447782cf7547ec6c6dba9c2377f0e7c71f05d2e3dee254b54c9abca8ff8fb` |
| `test_global_norm` | `True` | `1138880` | `0e87e829077b9d63a4407476e367a6d50c7299f749f33454bacfa06554bdaf87` |
| `bench_sm120_matmul` | `True` | `2373912` | `22bce600c4d9033455cf8788db03fbb5fcb33332c22283ba2c739dac5709e389` |
| `bench_sm120_attention` | `True` | `1728312` | `8a4c7c1c24d4cdd518803be25d39972076271575a94f886aa055b3bf875a463e` |
| `bench_sm120_layernorm` | `True` | `1233728` | `a53726712083c4556c2067318266ee6104d26843d8ca5812d763aa484864ac9a` |
| `bench_sm120_runtime` | `True` | `2221864` | `b6dd4a30434787f2fcf87fcaaa68185fdf3c692929fc35b5a397584aecf7875d` |
| `train_gpt2cu` | `True` | `3073832` | `c06e2ea64e88a9b668bfe445852fa4d9a0ca41ce2ab73194ee16fd66a709554a` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
