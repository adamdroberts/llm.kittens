# SM120 Round Manifest

- run label: `codex_sm120_cublas_dinp_fcproj_maxconn8_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_maxconn8_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_cublas_dinp_fcproj_maxconn8_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `646`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `4c8220822aaafafe50aaaf33db725d1f47387885f8313d4b87078e26d52e6d49` |
| `test_attention` | `True` | `1800528` | `8451b7537335a5c853af56ab2145be9f0a2bfcb7e2788b0c1110df653a5190cf` |
| `test_layernorm` | `True` | `1278296` | `73f6290312c62f7c43725954d7ab3547fa10c290e9bf18552de1d4275a02ef5a` |
| `test_bias` | `True` | `2089120` | `8ee450ea3e9a88fa0b9889a086b02431d39a80ce25ec440921ce0068d69af41b` |
| `test_gelu` | `True` | `1179912` | `a806bca5bca6111c5cee82a2c93ed6e2efc1a14cb1b39045e4495761c825ee83` |
| `test_fused_classifier` | `True` | `1208704` | `20ecf38e78b85d3919028f82231e15abec690597f7f2017a2de7cc0fdbe58da4` |
| `test_encoder` | `True` | `1210168` | `0363dc6716bd0b5f3b5f8899662d73d36381c8a8fd780d0f1ee3b51aa82a585f` |
| `test_adamw` | `True` | `1183768` | `ef9caeb9083cf8c38058aae0850ce20c3b717d68d7bc21602b62a56a3040bf79` |
| `test_global_norm` | `True` | `1179464` | `c6be7ae017918836eb79f0896a0f01aa3eabb61db77592d7e70765a0d9d2f4fa` |
| `bench_sm120_matmul` | `True` | `2410304` | `861a7c3891a6a8e68009fb7ae76303f7346a2471ab773a463404443706d67fd0` |
| `bench_sm120_attention` | `True` | `1768800` | `bd1eb830112a8cd4acc9e0a5b8051b0ecdcba2f079331b50b6e589b6441b48b6` |
| `bench_sm120_layernorm` | `True` | `1274232` | `d95a6caadfd7ba05cddb9079bd4f58fe4671a68f3e0835ab26cb4a9017c9f928` |
| `bench_sm120_runtime` | `True` | `2271168` | `e2f934b2621782b7387c4856230dd862bfa88c89b4ed391bfc2bb2bb6e57986e` |
| `train_gpt2cu` | `True` | `3105552` | `4cb18feeeab14db071c355f9429079f0f485e7e935c57b9b339bd7a47babbea9` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
