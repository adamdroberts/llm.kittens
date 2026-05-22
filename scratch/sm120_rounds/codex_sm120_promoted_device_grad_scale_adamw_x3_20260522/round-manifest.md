# SM120 Round Manifest

- run label: `codex_sm120_promoted_device_grad_scale_adamw_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_device_grad_scale_adamw_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_device_grad_scale_adamw_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `631`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `c3a50931c2151b956d7aae6d7f9a6b4022df3deb7515cf19875a77f5e8b15aa7` |
| `test_attention` | `True` | `1800528` | `aed6bdb2c5672d055e4810934d1e278d48d76e86fa5b7455f00d31c00df1b7ab` |
| `test_layernorm` | `True` | `1278296` | `a0860c28ac3bac8027f6b3f202a10b7d5552fe7f64adbbaecfa8ab2c2833815c` |
| `test_bias` | `True` | `2089120` | `bc6de8976a8185732171792ec30ec5b4b81b43c354930d246d5a61234783f8d1` |
| `test_gelu` | `True` | `1179912` | `51f96a13803abfbf11868892b1fecdc77d948693e30f8821183ae705d1fa225f` |
| `test_fused_classifier` | `True` | `1208704` | `a7e3a5f7a7a0e0848b48dabb3c6ba0453b1bcb189fbdfe3d046a32fcb6b1b340` |
| `test_encoder` | `True` | `1210168` | `600953d9b11907c094fd040197e1d7b1314f382dd2fb850df128c9d13d69da0e` |
| `test_adamw` | `True` | `1183768` | `7377e7486979c7072e2e27a4033f33240f57a3e87d03b8a0d6f60055bc7221b3` |
| `test_global_norm` | `True` | `1179464` | `eb4bac0e31a12f08a0f1c3c0781b7e69884b17f35ba7056ac33794d22a458164` |
| `bench_sm120_matmul` | `True` | `2410304` | `315d2af490ae5fb3787952af372a445a4f9b20f5e77333e12946fb29b5503040` |
| `bench_sm120_attention` | `True` | `1768800` | `940774589cea7cd3359f1ec39e7ac7c746ec2a0fde7fc2faa057be11cf451e33` |
| `bench_sm120_layernorm` | `True` | `1274232` | `0978fdd8227ccf866aeef902f66b98bedfc9f5f1559a0e22208a3bcfa0f4d250` |
| `bench_sm120_runtime` | `True` | `2271168` | `ceb0f390f5686d46827f5d5172a103014c77856e976ab3f308b8717349efd30e` |
| `train_gpt2cu` | `True` | `3119192` | `636da218083f11c6a79855e7e7a349c27c7e1b605ae27eda7c506889933ec9b6` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
