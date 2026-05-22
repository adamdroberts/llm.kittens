# SM120 Round Manifest

- run label: `codex_sm120_default_after_combo_rebuild_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_default_after_combo_rebuild_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_default_after_combo_rebuild_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `514`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `e2c728c1d7de6d2d66cfeb2cd4e48e4a3ada72d8e748253829bcf476883728a3` |
| `test_attention` | `True` | `1760032` | `3fe0a8373fdaac4066bed847acda09ddf9129b329b04c8213308e9a5da419a4d` |
| `test_layernorm` | `True` | `1237784` | `08468bf0a337b69ffc4810ec3f7f9ef8a46707a7456ab5eabfe48b1ce60053f4` |
| `test_bias` | `True` | `2048616` | `f4052ca13c8dafd7749ee6ac08486de918e1f749964e3b6b1b9a1b0d418bb0f3` |
| `test_gelu` | `True` | `1139336` | `141a935ecf9f973f248cd22c0a234fe86d19d63a03783a8fc9259e528dd59e5c` |
| `test_fused_classifier` | `True` | `1164032` | `4667e6f1375ade36a80633e658d5816ee31e80bd33785865abc09b37aa15d539` |
| `test_encoder` | `True` | `1165512` | `57c20956500a82e32531607da93aa432ef87c4e73b564164b75e9c4672cd2790` |
| `test_adamw` | `True` | `1138408` | `3f8c22fa74279669e2a1683556fb1f5b2533c13957c15927173bdeac77882a15` |
| `test_global_norm` | `True` | `1138880` | `71b43625f559cf5d4c88175d8429714e7dca748192978f6caba644e6a4529a40` |
| `bench_sm120_matmul` | `True` | `2373912` | `0c5de63398e69a00ed4e162a44a74ee09a0d09b4d8bd73d31522887e24b47dd6` |
| `bench_sm120_attention` | `True` | `1728312` | `527a625e4110a3d254ff7e386950fbab45522370f3c97271dc6e163acfbe7514` |
| `bench_sm120_layernorm` | `True` | `1233728` | `19e0b4872e5ebce62130942e7267eb1b3e4252f70a4f567ee2d39b54abdb92ae` |
| `bench_sm120_runtime` | `True` | `2221864` | `c8dffa1e3bc804034fa017eed8d6bc7ab38cf123590cc462d7e5186b53c11605` |
| `train_gpt2cu` | `True` | `3060032` | `9ec1cbc35e7cca7e2ddb99927ca83ab61047b8fffb2722e2774e40c5e8433894` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
