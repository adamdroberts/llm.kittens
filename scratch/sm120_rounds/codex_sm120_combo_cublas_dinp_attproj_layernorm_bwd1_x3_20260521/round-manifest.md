# SM120 Round Manifest

- run label: `codex_sm120_combo_cublas_dinp_attproj_layernorm_bwd1_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cublas_dinp_attproj_layernorm_bwd1_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cublas_dinp_attproj_layernorm_bwd1_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `516`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `a18ee7b2eb499cce4d49918489866a91d689eab74f0b0eb6048dec4067337822` |
| `test_attention` | `True` | `1760032` | `2f7696503c3d342d204006292a57664d72f7735d6817b78b69b007a3daff4a51` |
| `test_layernorm` | `True` | `1237784` | `f60966653f38ed20292456c1ae598ceffd4d4fd586b430cead844c920bcbe1b1` |
| `test_bias` | `True` | `2048616` | `de33746d5dd609d171e35c80df6bcea2e02b3efd1774eb3b6fbc0c2aefa737a2` |
| `test_gelu` | `True` | `1139336` | `8db86f22aa44dcc133f85923c00fbf48a1d49d8eca76cbc18896cc898271fb74` |
| `test_fused_classifier` | `True` | `1164032` | `1a13aa767c235ce3d383949c932412750654fd51f5491c2ce0523937bae4e670` |
| `test_encoder` | `True` | `1165512` | `304775e51f7816b3fe6599aa2fcc3bc98a0cd2dcd458f8d34ffb234cd1429ded` |
| `test_adamw` | `True` | `1138408` | `8072f09247f5a42f56d0b985dc6ffffbe6920323115159b7f60bb7a4c357ca17` |
| `test_global_norm` | `True` | `1138880` | `f0054c5988be2ed1c017059942d1d75d495b8fa4fc76826085acd680cd384201` |
| `bench_sm120_matmul` | `True` | `2373912` | `4d61d49a34a6cd5ec54e75eb4091d9003519803bdb323e4a025ea2c1db84677d` |
| `bench_sm120_attention` | `True` | `1728312` | `739c24205ce33a39d8780414878c52c3e7600ef09663b13814b4ac98557e3d88` |
| `bench_sm120_layernorm` | `True` | `1233728` | `f7d619dd4728e1167c9a2de41646546160a3ccda6667c978f9bd0f2028f3a780` |
| `bench_sm120_runtime` | `True` | `2221864` | `41db8b5e653af569922bdb22a93db4145f5d5ca451f7b700eaa7bafc2fe911f0` |
| `train_gpt2cu` | `True` | `3060032` | `1e31d83a7df92b13c03f27d4ac82cec4bd2e5a36a609f8a2f76085c4a55ef6da` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
