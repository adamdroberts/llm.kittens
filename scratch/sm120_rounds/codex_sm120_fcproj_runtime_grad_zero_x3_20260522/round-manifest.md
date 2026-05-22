# SM120 Round Manifest

- run label: `codex_sm120_fcproj_runtime_grad_zero_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_fcproj_runtime_grad_zero_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_fcproj_runtime_grad_zero_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `665`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `bbe35b663751a0ae64e2a533ffe8b0f2fec27e7c30a4fc4488efbf430505512f` |
| `test_attention` | `True` | `1800528` | `574277f88ed7a6aa981e056f1042f68a9282bd9d9bb82a63e677568693b6bef9` |
| `test_layernorm` | `True` | `1278296` | `54320acf18da6aa9623aa3271962c41ea6a756205d0132646ac748b05d4c641b` |
| `test_bias` | `True` | `2089120` | `9f5d27e5773e931ad80d67bdf3d86e485893d03564cdb311264ecb4357796bca` |
| `test_gelu` | `True` | `1179912` | `d3775d7cbb123ee71f3319ba2b289524b6a643b3def53cdd978290a522c93e2e` |
| `test_fused_classifier` | `True` | `1208704` | `50a6733d0b598876a04c047beff57e93031f41ebdce1c2cf4fe55b6e4497abbd` |
| `test_encoder` | `True` | `1210168` | `d65000213353c07510ddcbd79c57a1474a19c219d68c70406c37308f9942d9db` |
| `test_adamw` | `True` | `1183768` | `1ab2c1e0940b5b8430306d46f64e1197c29ad6f954af4b8b193b9e08e5e73511` |
| `test_global_norm` | `True` | `1179464` | `9d1bf3cc127de3c2275fd0e6900dd7296c88d577b52238150bf3a10b08059f20` |
| `bench_sm120_matmul` | `True` | `2410304` | `77b85594dc12e8b54baafc69a6b40eb1b04e9e92d1821e9005edc7cb03c412ad` |
| `bench_sm120_attention` | `True` | `1768800` | `59ee4c35c8af0be2b423fd8a5f0164b827b89027d57238441721f2f289894470` |
| `bench_sm120_layernorm` | `True` | `1274232` | `dc50d5a323dd092ca5d25494ad25c59a44551aedb5a6a88f1c37c7e77f14e9c7` |
| `bench_sm120_runtime` | `True` | `2271168` | `2cff98c4fadeeaf988c08c95bea39adcb77f974523fa21eea015d2ace30d5ba2` |
| `train_gpt2cu` | `True` | `3105552` | `234141d3bdf56d84e89ece5e8ec3decf5a986a3373a6a3b0608782ec2130885d` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
