# SM120 Round Manifest

- run label: `codex_sm120_promoted_attn_fwd16_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_attn_fwd16_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_attn_fwd16_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `638`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `b2d5293d2e7e6b7d6a8506d4de0a51ccbab4750f6b14ecc1d235da5ed0840513` |
| `test_attention` | `True` | `1677448` | `6c46e75b96a171e2f48a624533d9ebc5c133253b94c53655f6e1eff613608b8b` |
| `test_layernorm` | `True` | `1278296` | `d60bd579c1364ac6c7cc8afabcbc304e8a68078972c45e8272490b64f374bf9c` |
| `test_bias` | `True` | `2089120` | `5cb2d49a6942cba6cca76501bbadef20f9372342f75e32bfb384ab0b158e4ea3` |
| `test_gelu` | `True` | `1179912` | `bff38d09425c280fddece987c43bab29af3f3f3856215b3e827278ae3016a8bf` |
| `test_fused_classifier` | `True` | `1208704` | `a16ca44f64980b6c10ea8c26f6f2321062892fff6b28a6b7fea968929ada6a08` |
| `test_encoder` | `True` | `1210168` | `c8f14723626cf2d773a008bc5ef0060a03d7cf2a172ab182bdf68ff1d953d35f` |
| `test_adamw` | `True` | `1183768` | `984f052adc2138f6f521c49667ee29b010563f62e17fc1cf23e4f5a28d7f5d0c` |
| `test_global_norm` | `True` | `1179464` | `74d9815cdf87897557660d7d5c2f9c9a07f3c21ed72b02b64b4f212197e77041` |
| `bench_sm120_matmul` | `True` | `2410304` | `1ecbb80d998d7e1b1c90d199a8f7f2ee90769babd3208b81093ee016925edbb1` |
| `bench_sm120_attention` | `True` | `1642168` | `fd0092f510588ea400227b0690f4bf23bef8bbee5de3a156b62a61e945bd52bf` |
| `bench_sm120_layernorm` | `True` | `1274232` | `d9647fc7cbc54c895798c117e49dd7b326411124531a16395412f0b5db0b63cf` |
| `bench_sm120_runtime` | `True` | `2271168` | `122025259eabd7cb2269d941956979bb8df15687d700066bb61470ede5f594ea` |
| `train_gpt2cu` | `True` | `2974536` | `281dc5e6cd37cdb5c9b06b024e1513e38312877f2a75bf823e285eaf8b37821c` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
