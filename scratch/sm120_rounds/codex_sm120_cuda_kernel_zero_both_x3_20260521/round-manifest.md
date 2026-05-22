# SM120 Round Manifest

- run label: `codex_sm120_cuda_kernel_zero_both_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_cuda_kernel_zero_both_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_cuda_kernel_zero_both_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `528`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `4f0a628b91968ed9d12c3f5161033850ee746e72a7a9ac40d40e5c2ea24543f4` |
| `test_attention` | `True` | `1760032` | `9ed0df05a1e3ee686ac43eee9f6d2d172849c93106f1206fff44a1212afcd839` |
| `test_layernorm` | `True` | `1237784` | `58034c94cb5a97d2b25f51321b65779de39947ca0c759187557acbd9fd5093e0` |
| `test_bias` | `True` | `2048616` | `cc96ec713b7576cfb5815a2ce7e5957dc3aa28423118cbeb2c32a188ef7fb2d8` |
| `test_gelu` | `True` | `1139336` | `b35b27e1085f21a086b086cad2a0810bb314cf291ac185d3cb41322a945ec1c5` |
| `test_fused_classifier` | `True` | `1164032` | `61552242ab227fbb17adaf18b74a72d213e961abb8fe99c6282089fa09cf7b3b` |
| `test_encoder` | `True` | `1165512` | `7e530c4c46daa8521335cb59f6ceecd65b9f7491fbf9efcf5c6246e1441f0086` |
| `test_adamw` | `True` | `1138408` | `e70e0be77a7fec105c034d93f9178c54a07068e4e7531dac5a3eb5b58dd00503` |
| `test_global_norm` | `True` | `1138880` | `00222ef265dfa3d263ee454a4abf204b87a5a891571ef8f874c20d5644c7c3d8` |
| `bench_sm120_matmul` | `True` | `2373912` | `76c2a22972da1667aecb52c90b8769de633c190f7db16baee14e02f1d272d5ff` |
| `bench_sm120_attention` | `True` | `1728312` | `be58e52ea518084a826b6db215bb58e4813a99eed522e7861763de3630b005e5` |
| `bench_sm120_layernorm` | `True` | `1233728` | `fdaf5a39dfef856aff7082223d0976b758d0b3a7e480a75bf8b3f907a3150942` |
| `bench_sm120_runtime` | `True` | `2221864` | `86ab4d527b0cbdc4f57ea4767d98d2fbc18eb8695ea1d937ebcd335ad3603ca2` |
| `train_gpt2cu` | `True` | `3060032` | `0ab32d73ab6e01473b5f8fb95396d4bc8ba5556f96d3927be5df124682e67aee` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
