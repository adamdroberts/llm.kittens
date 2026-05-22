# SM120 Round Manifest

- run label: `codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `596`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `a30e9311ab96d8dca5d5e47928373548eb4bc4b3d05bf8c013e338c8b6fe854a` |
| `test_attention` | `True` | `1800528` | `98f73731cbc3d17462fb5589e35c21368274247eb23639fd7440c140cc87ad22` |
| `test_layernorm` | `True` | `1278296` | `1478bebb032ab6595c8776eb659b92281a24a42849be43e06b432920534703c3` |
| `test_bias` | `True` | `2089120` | `6c7a381edcf6da42ed16c049b29b509d1e93efad481977b42c999a8307a0ed16` |
| `test_gelu` | `True` | `1179912` | `ad09f9ea9b4373256fcb1464ac07bd05704c2ca04f6d3453b47ae5cd820b2b1a` |
| `test_fused_classifier` | `True` | `1208704` | `08f92626592e6491380da9872570d3b187bd3f87a89ce337a5285c793e3ac9cb` |
| `test_encoder` | `True` | `1210168` | `1c58e99eb77246bf98e89305f80779f4fb7ca185806a9c7217c40375969becf1` |
| `test_adamw` | `True` | `1183768` | `20f8e409124906fa10f7cd27cefb0f910628655b7d8da77877bba342006da2c2` |
| `test_global_norm` | `True` | `1179464` | `5a2cf7eede6beb82c0a15c889b63c234ab311b5f248a041d46913efe1643b398` |
| `bench_sm120_matmul` | `True` | `2410304` | `997f64cbd37f73aa90b35ffa1a4cc5e7210c531564ce9e4edbeead8db8fef634` |
| `bench_sm120_attention` | `True` | `1768800` | `5730da5a25176fde07dad21ca293164234680bfb4be8fa82bdfae16a4b637314` |
| `bench_sm120_layernorm` | `True` | `1274232` | `ebf74f56493051f264fddf5c26bd15f61696aa7b952e73fa54b4bafb2763a46a` |
| `bench_sm120_runtime` | `True` | `2271168` | `324039b005ce020475775c482b6375e0a20c97980dda551c42d6237ebbcf3419` |
| `train_gpt2cu` | `True` | `3105552` | `72254964f04d1fb58669138275f9509ebd899949c8f4ee0a963ae0abc9ffdbce` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
