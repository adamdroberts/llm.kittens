# SM120 Round Manifest

- run label: `codex_sm120_runtime_grad_zero_current2_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_runtime_grad_zero_current2_x10_20260522`
- device arch: `SM120`
- max steps: `10`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `687`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `f4dc4b7aadfbf96f8ec7627adbb91b6367ddab9a97b8f21e005c45e5fce6ad5f` |
| `test_attention` | `True` | `1800528` | `c15c44a4ab1369ba6f2b784d3df2d3024784d53290e0036a46782a89f8cd758a` |
| `test_layernorm` | `True` | `1278296` | `4aa3b46b3e48b1d82ff1acbb979c8c22395652eaac9434a37c867b699845ceb6` |
| `test_bias` | `True` | `2089120` | `8f3dea2ddfeb70018500a684f7d4986b2edb2c2e422996088d042af2aa095d1d` |
| `test_gelu` | `True` | `1179912` | `6b39266509c2c45024f6e7dd42a76a2a009ce733c4057b82f6f7fae9f49c3ded` |
| `test_fused_classifier` | `True` | `1208704` | `e0da2c3fa46abc21111f9e4fe7db0a423c9db2ccb269716c651f1abf9c32855d` |
| `test_encoder` | `True` | `1210168` | `4f77803050df75a8eb9d263dce90350676fd3a15c7e742073c92502ca9d36952` |
| `test_adamw` | `True` | `1183768` | `31d55f3376c3740186e301298bf951684031e83c1060acdc9c1b68873d2532b7` |
| `test_global_norm` | `True` | `1179464` | `1d127008e1396fc29ac5d2b4c12b01bd3d4b739d6648e91dc0a7dfed4540eaac` |
| `bench_sm120_matmul` | `True` | `2410304` | `298138ea51099617d86704b247cf939a6fde238285c577615ae1e1b2f414c0c0` |
| `bench_sm120_attention` | `True` | `1768800` | `18352855f87e71b92a3235b90b8f73601bf195c5ddada62232ffe0133218b932` |
| `bench_sm120_layernorm` | `True` | `1274232` | `4d326770f4de6f668d77f62339a26da982bcf3558484913ed59671bf27d8c851` |
| `bench_sm120_runtime` | `True` | `2271168` | `a5f630de0adc30e4a49d8a6f9d2c720b5b6c742d5b6cb6bd1a081b76685bb853` |
| `train_gpt2cu` | `True` | `3105552` | `21a34b8db225946d6a483311c40de83b7cb128673f9e5599c6dd1686d06c1336` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
