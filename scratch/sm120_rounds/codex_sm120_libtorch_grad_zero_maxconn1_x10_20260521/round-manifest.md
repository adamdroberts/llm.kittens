# SM120 Round Manifest

- run label: `codex_sm120_libtorch_grad_zero_maxconn1_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_maxconn1_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_libtorch_grad_zero_maxconn1_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `578`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `f65566d6ea8f20a066b4205abf8992ba78f8c6f5f21e7e2398fafeed2778073e` |
| `test_attention` | `True` | `1800528` | `2e7ae60f40823db96854db4b334225a69b5b7e8c601e5a0b4ff5c4f91ea697ab` |
| `test_layernorm` | `True` | `1278296` | `de451368d689aad6b80e3a4cce02aa362c356b0db38975c64a5cd3f63ec2ce4c` |
| `test_bias` | `True` | `2089120` | `1c00c1585972d07e89e2568cc8e5cd2fc0ca3f06316197656eb34854a052b015` |
| `test_gelu` | `True` | `1179912` | `cdeef44a8076924ebdcc9dac1a31ddec5be2e0d9232414b6d8edbe67aa65a71c` |
| `test_fused_classifier` | `True` | `1208704` | `cec7aaf32185075ba218f780e2c01164357ae22a04da8af46a2c15568795a836` |
| `test_encoder` | `True` | `1210168` | `0b5bdfa039c1d05a80a6e6517cd82e9d220710d8f6c93e3139c14095287a790e` |
| `test_adamw` | `True` | `1183768` | `c9a62463574c5d41738283d3ddc1a558a038de870bd3766af52a64badda993b0` |
| `test_global_norm` | `True` | `1179464` | `81c85fdc355f29c607790b74c1e196743c7a20dfc013b8c33cfefa3f89141432` |
| `bench_sm120_matmul` | `True` | `2410304` | `1a76fdd044f1c4731f9022949862cfab832c3c1911dab4265ae78332719a8bad` |
| `bench_sm120_attention` | `True` | `1768800` | `ec31fd66edd9687f93afc89efd6e53f96ed547af0791074596b50283b6586e08` |
| `bench_sm120_layernorm` | `True` | `1274232` | `bff1697fe058bf5ba0a15d4a494f97e3409cd1c932a9aaf1549a326891fe6b46` |
| `bench_sm120_runtime` | `True` | `2271168` | `99c0c06745aa5dce48bbd0a42306c1052716104e50ec37292b79be092034a26b` |
| `train_gpt2cu` | `True` | `3105552` | `cfc486925c97c78259ab2b014dd91b0159e8976df4f9a870e947136dfd3a3299` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
