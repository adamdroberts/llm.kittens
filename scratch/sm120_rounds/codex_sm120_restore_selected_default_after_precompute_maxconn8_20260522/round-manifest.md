# SM120 Round Manifest

- run label: `codex_sm120_restore_selected_default_after_precompute_maxconn8_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_selected_default_after_precompute_maxconn8_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_selected_default_after_precompute_maxconn8_20260522`
- device arch: `SM120`
- max steps: `0`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `657`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `70bc66607ea04a64b86b0427326aed1c59067f07c283499702699f07dcaf9113` |
| `test_attention` | `True` | `1800528` | `76e28e04e29077afb4186d3c69a9b44bc4298a0fe4f2fb7ed15749278880e75c` |
| `test_layernorm` | `True` | `1278296` | `6211e44b552e5bd2e1554066e1955dfdc3f4bdea447f19e0b1cf177eb71f92f7` |
| `test_bias` | `True` | `2089120` | `4bb41432f76bd63eb3a5bc348a1554aefa3e9ea0c0470dd610c98302ed10f3d5` |
| `test_gelu` | `True` | `1179912` | `44fe6e5165881b7e28f74070e1f7635a8a5f406bc5e6000b44ef852a6fbb13b4` |
| `test_fused_classifier` | `True` | `1208704` | `86621dfb6bedc581a12ebd4288b1a79b5899b699d0e69a1f2b89f87fa977fa2f` |
| `test_encoder` | `True` | `1210168` | `b9aa3cb994639a937ec3d9c3fb28782d631920ea5539ee54115f097c5728a4ee` |
| `test_adamw` | `True` | `1183768` | `58b0d46ff215f4e4a6595a7fc01d89b4ecd92cb139e7dbba93d5285dade669b3` |
| `test_global_norm` | `True` | `1179464` | `d6f4958a9b4a3866e2ababf3f08019762048f1eefff8cc80214e8159d40ea5cf` |
| `bench_sm120_matmul` | `True` | `2410304` | `228bee283d88d1cceae751e72daedd7e4a5b1769f1439b5ba9daeb44c8d0790b` |
| `bench_sm120_attention` | `True` | `1768800` | `f29d94f6b13d2487d8007129c0fdd275e2a45d31437a0846d008d0470944c147` |
| `bench_sm120_layernorm` | `True` | `1274232` | `41c433f0cff10522afcfbb7a08ecac441cab3ba335132090b193d44e21176aea` |
| `bench_sm120_runtime` | `True` | `2271168` | `8c52018ddffe31a460162be4dc58b807d2f63af0166198332d832988443c66d7` |
| `train_gpt2cu` | `True` | `3105552` | `a9f1277e26ec19923ea838e39dc7c26f7c059710c9cf35f6cc1c613c59c69e8b` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
