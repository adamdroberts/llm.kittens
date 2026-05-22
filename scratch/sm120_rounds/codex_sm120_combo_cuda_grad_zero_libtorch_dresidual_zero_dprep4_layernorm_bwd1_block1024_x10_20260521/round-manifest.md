# SM120 Round Manifest

- run label: `codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `597`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `fe10f2a5fc3f36cbcacd7c0459c56634af7ef2b927c50b64b9f713f5acd4a55d` |
| `test_attention` | `True` | `1800528` | `248dc49e1ee0a81b5fbf4e6eaff95cdacba2f0a432471f2024f109bfbf4362c3` |
| `test_layernorm` | `True` | `1278296` | `4a7cb41d633dcddd3be351ac02c7797dcf407602cd4ba76a84b5e9cd173fb4cc` |
| `test_bias` | `True` | `2089120` | `4b8f226565b2e7016b4c29d3b683fb3e21ca7ffaf3aee3bbc10baafcf9e68039` |
| `test_gelu` | `True` | `1179912` | `d3ab8ce71658affbf28d3c31de609ae62d4486fa482c5824f64e83e8682f8286` |
| `test_fused_classifier` | `True` | `1208704` | `1a42664831c4078f3e4643139e41725bf413ad7363821d0ae44395558a78b9f6` |
| `test_encoder` | `True` | `1210168` | `84a9ca2defd11abc454348f02ff27c27cb8604d7492e94a1985d530e4019cdfb` |
| `test_adamw` | `True` | `1183768` | `5ca531acfc544cef0817d672ec22278f8942c56be8032f6d445dfd86682d779a` |
| `test_global_norm` | `True` | `1179464` | `1385cc6be09a22004691f4be8e37514db75a98d3c8324b0be78b4a520f23b92f` |
| `bench_sm120_matmul` | `True` | `2410304` | `89acb6a0fe32c820f9b5bf2dcc82aa6b23fd5bfe5c308d53d40adb5ae95f7fdc` |
| `bench_sm120_attention` | `True` | `1768800` | `5671031b44923bde5566dfa3b05565ff0a72560c16ff5e2d7d2c54ad30f6684b` |
| `bench_sm120_layernorm` | `True` | `1274232` | `0c6de28744f1f2cc8edf03ec43e6cb641193c881eea0421214418344fc9ec325` |
| `bench_sm120_runtime` | `True` | `2271168` | `6320118ad931a4891ef57b7a7818dc89dadd10fd584e9b6c95cd22599cd05c55` |
| `train_gpt2cu` | `True` | `3105552` | `8a1f37a6b67ad3aabac298593c1535a4f9022d0a0f7639766d31a9713a3aa74e` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
