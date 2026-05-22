# SM120 Round Manifest

- run label: `codex_sm120_promoted_cublas_dinp_qkv_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_qkv_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_cublas_dinp_qkv_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `625`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `23f63fc3b9830f52eb553f8f9304f2e73f2e40221df1cdbc9733ef862f7fd915` |
| `test_attention` | `True` | `1800528` | `626999910b4a8e5f4eb153d66cf0d860e0b44b57fb1e28d15770810d69b2aa6f` |
| `test_layernorm` | `True` | `1278296` | `9fd7d6d47b5290f0468e556614ee70d67e17d04a8649da6ebe8afda25a1eb755` |
| `test_bias` | `True` | `2089120` | `6c8b79f0ef786956d9cec031d293304fd86ad5233d65399abdc4f450eb2991d9` |
| `test_gelu` | `True` | `1179912` | `e6265a8d05d60fe9ebd07b17ac6119e3c64a5a1e7244aad1038baa3e266988b4` |
| `test_fused_classifier` | `True` | `1208704` | `b10da392d77448c5535fdfb582ac2da58df240e51b9f0a97ede182bdfc07aeeb` |
| `test_encoder` | `True` | `1210168` | `86be5987f570e2c569e2deb779359b6e2b156579475a8afed61b509cfc85cf16` |
| `test_adamw` | `True` | `1183768` | `3e1086a6dff5cd0906b13e03aa5dcaf65826fbb2b161b48f3165d21e7a320271` |
| `test_global_norm` | `True` | `1179464` | `ce0366319f30a6fa8a039ab29af687a304984af584363759bb5945c45ea025c0` |
| `bench_sm120_matmul` | `True` | `2410304` | `8e2513e5805cb9eff4612423f273fc67daa16b6603a6ba9ccd3e339bed2ea512` |
| `bench_sm120_attention` | `True` | `1768800` | `f5efa49cc3ed9f2bd284795144bdd7e7887bf89959c2467209b941ac7c9f8a01` |
| `bench_sm120_layernorm` | `True` | `1274232` | `481e9e8f548ad96d7e558ea4a2e53f9dc3018f3def24837ad52be7301fbc3468` |
| `bench_sm120_runtime` | `True` | `2271168` | `48c398cc535227ff95c4883a097d375b7b9982c1a3f0a7aedcc6713270569461` |
| `train_gpt2cu` | `True` | `3105552` | `a177b778de5343db8be9c12959bba277a4875dd5d708e6f64ea8de78eb31139d` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
