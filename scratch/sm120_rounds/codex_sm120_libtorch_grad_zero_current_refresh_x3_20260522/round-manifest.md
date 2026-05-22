# SM120 Round Manifest

- run label: `codex_sm120_libtorch_grad_zero_current_refresh_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_current_refresh_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_libtorch_grad_zero_current_refresh_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `681`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `d52539c6bfc593d6300203fb344c116517a5e9bb1f762609a93ff871a25a4be5` |
| `test_attention` | `True` | `1800528` | `f0e6741a2685511d0d28926562894a4272bbc31c5ee1cb479ddc76b4e10f8447` |
| `test_layernorm` | `True` | `1278296` | `d1efec671ea3b4144c3f3d2b9766bda9b70fdf45ac6a9f2c66b73ecc493667e2` |
| `test_bias` | `True` | `2089120` | `eb1457589410694dacdc5e5d9cff597e554a7242fe604278d196a507de24a896` |
| `test_gelu` | `True` | `1179912` | `89658cda76bf3e0471fc9d296de66d8c69ae721b93559c5b30b99e13f6ed15c5` |
| `test_fused_classifier` | `True` | `1208704` | `12208b8e1d23fc524be9b362551cc2f8f29ca3253273defa1b2607b67d5ca48f` |
| `test_encoder` | `True` | `1210168` | `074c8b919d8f0c79d376a1c912f3d52bc0224f5b8fb344b23391ac3ad1baa396` |
| `test_adamw` | `True` | `1183768` | `c1497f8cf09b1e0c2047ff6769a98190ecdd511921a237d0d61af12e5ec1c15b` |
| `test_global_norm` | `True` | `1179464` | `7489bd55ef8fde8cd692124d9be78fa727191c83fbb60a6d725602dc79ff0ebd` |
| `bench_sm120_matmul` | `True` | `2410304` | `26f4f5043088cf724b3ce722aac5c8057506857f0d30d330688d6260f25d0395` |
| `bench_sm120_attention` | `True` | `1768800` | `e0b8ea2c76edbc0eeaf5abb543d5465e4e0ebb62a7da85c5eb0a3b640d4e0aad` |
| `bench_sm120_layernorm` | `True` | `1274232` | `8a1e30215cf460072dae0f9e0cb4ce198bed4e68e0da103d0e25beb975bdf423` |
| `bench_sm120_runtime` | `True` | `2271168` | `f108e6c6a5ad531c2c075fb66d5cc7399cdb40316f3931ecdd8229eb08670144` |
| `train_gpt2cu` | `True` | `3105632` | `89cd4ad9606c79abe4f927cf44deef08fd169a7da913f62df821e7d21c8aa9f2` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
