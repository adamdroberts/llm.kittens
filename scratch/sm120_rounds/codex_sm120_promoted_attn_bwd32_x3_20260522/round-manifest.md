# SM120 Round Manifest

- run label: `codex_sm120_promoted_attn_bwd32_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_attn_bwd32_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `639`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `78ed744848d79fa5544b3991347796da0dd6b0c1e6edcbe0c24cf1466459a3f1` |
| `test_attention` | `True` | `1910920` | `86f79a70d0632d07ad7edfc7f22b82310d639a0f24d6c554d7ca99fe00466f9f` |
| `test_layernorm` | `True` | `1278296` | `75d26ebe5b56350ccbb09cd4706b417a04e9f2421cf605b4f2233b949abdd936` |
| `test_bias` | `True` | `2089120` | `640c6cc63332d3cc48239fc2a7a66dc9f440f8bb2e8a4acbf1b564195b5481b3` |
| `test_gelu` | `True` | `1179912` | `f95f8b1cf57f869d68b6a7708a0f892f52a0224783f1ded2f7f1f822c6a55cc2` |
| `test_fused_classifier` | `True` | `1208704` | `ed40f7636dc1b5974d6ee53d62b9b96b44baca2eed3ddb55d27ffcecbc109a05` |
| `test_encoder` | `True` | `1210168` | `f7030e60f29174b8ce2d79b347e0bfce288212570a3487e5197dc55c1139d803` |
| `test_adamw` | `True` | `1183768` | `8db7a4a9affec16599762f83e63dd00c16e22b1aec12b535fa0175c4ad58c938` |
| `test_global_norm` | `True` | `1179464` | `7c1a50384423de290330d5190575525b8382bdd36c037177f9ce891fe6b17e0e` |
| `bench_sm120_matmul` | `True` | `2410304` | `c1a44c88a023a7ed3287dbca01e87952ad4fac4fc942d45f2ba45be272040223` |
| `bench_sm120_attention` | `True` | `1879736` | `c9d541448f1aa63c6134e06522f06b7f9fb5db194715107dec8a9af1983bbba0` |
| `bench_sm120_layernorm` | `True` | `1274232` | `80d96b423d161e01c75ac89c98f9e79aaa3f74deacc0ca0e9aacff92e0708b26` |
| `bench_sm120_runtime` | `True` | `2271168` | `9d6be8b1eaf5f255af9f0c721c93d5c7b3c03ab630544814f0d4d7e0b8eb08a0` |
| `train_gpt2cu` | `True` | `3212104` | `d254931734f4b26360e258efba5f65c1ffe5929fefc2d70d3ea567e5053cf8be` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
