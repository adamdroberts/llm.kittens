# SM120 Round Manifest

- run label: `codex_sm120_runtime_grad_zero_recovered_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_recovered_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_runtime_grad_zero_recovered_x10_20260522`
- device arch: `SM120`
- max steps: `10`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `662`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `3a0c4a99e848ee361547039affbc284eed1d00bf85b89da892bac12482cb30e3` |
| `test_attention` | `True` | `1800528` | `6283bd01a49a336a3b97a10ccfef9ddc26d2621bc7c609a070fcc3199cedfa2c` |
| `test_layernorm` | `True` | `1278296` | `5b5145915c974062f5e3d2d68cbcd48232632d4e42db1b49c691b9b053e148d1` |
| `test_bias` | `True` | `2089120` | `7912c4ce6186c2c3c7b57b4669f7011907769121ab62abdb6253dfa6732f6148` |
| `test_gelu` | `True` | `1179912` | `e46d85932eb1662c12e73b74c0e1a19fe957524e5546b804e453812129ea0d15` |
| `test_fused_classifier` | `True` | `1208704` | `2c3b2401d99f5644df23f210d36c0043e386c8ee4d1cd6ea4c82fe0c47b98cb0` |
| `test_encoder` | `True` | `1210168` | `443cdb76fd4b86867b09fb63dd215165d049a43192bb9255521fbc2ff73f3f49` |
| `test_adamw` | `True` | `1183768` | `5c7ac7bae16b9ec6ee1ef1178e9d95dbc9c1ee05c960ebdadacc9fc5a241bf43` |
| `test_global_norm` | `True` | `1179464` | `14df4fe61965b0e3659db26df132cf36e52aa67c3dea2060ef14c25f2140fa34` |
| `bench_sm120_matmul` | `True` | `2410304` | `30b165a738680f1b54d1a4d2b93a6e674f4477a031072a876abbbc7289810d19` |
| `bench_sm120_attention` | `True` | `1768800` | `305bcd97f769402153fcf5662641d85eb5d4762f481f793e46ea70bf8d0f7fc9` |
| `bench_sm120_layernorm` | `True` | `1274232` | `37ac8f8b0c8affb5780ddf61ccd52c66e3fa203090b12947f911329fff7579cf` |
| `bench_sm120_runtime` | `True` | `2271168` | `24d9c369f321a2e2ffa4c5a7ae527549fb778ca74b338deb261b23203ddb2776` |
| `train_gpt2cu` | `True` | `3105552` | `de8ef2a1244266dbfe851c21fb21b158b4653487356c142f2f4fd04ea3c3bcc0` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
