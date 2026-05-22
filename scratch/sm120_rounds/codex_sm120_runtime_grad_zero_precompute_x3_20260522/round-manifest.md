# SM120 Round Manifest

- run label: `codex_sm120_runtime_grad_zero_precompute_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_precompute_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_runtime_grad_zero_precompute_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `672`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `12c0a1e47dc7c1763e627dc92cde4e1e7968bef5a397108fc99966208895ddc5` |
| `test_attention` | `True` | `1800528` | `a4806a33ca1cbcdab1341b75d7a70a5c5682df9df1189db3c446831cc6be2b0b` |
| `test_layernorm` | `True` | `1278296` | `9949303bea4083d35771a3fab59505598cf38ccd8a855fab303754cf672320c4` |
| `test_bias` | `True` | `2089120` | `203023332582f0231f7e551fa1eddf494358048855424bf31748f1a2cf48c81e` |
| `test_gelu` | `True` | `1179912` | `1411a5a1faf2a5a0715d650a0b9406deb332e3dbe0f947b4ae4a24b0a1678d5d` |
| `test_fused_classifier` | `True` | `1208704` | `ae177a2125b1046caa19b1099a4aecb7d1f374d2106f69c1e05e6bd38d6ea7e0` |
| `test_encoder` | `True` | `1210168` | `eb2a02fd86a8e247ad5534f1d5eafcb8b6a0a36a5dee095d6d17a6858c552c4b` |
| `test_adamw` | `True` | `1183768` | `d801b654ea732152efaafa3059de58d8949a862fc2ba8f9dd88008891c86e3ef` |
| `test_global_norm` | `True` | `1179464` | `923aeb86da9b208e32f9c37e4f3ef0b369166ba758186a018243e2791ad38f54` |
| `bench_sm120_matmul` | `True` | `2410304` | `d96487f2f12ceeb6fea9e4691e8807c4a8f4e2926141e03e0c848a0040900974` |
| `bench_sm120_attention` | `True` | `1768800` | `341f67f01e359017ed3c38fbcaae98dce5e869cc94b400343b587bd39c967e9c` |
| `bench_sm120_layernorm` | `True` | `1274232` | `6d5047545aa25f2354fa1c146fb00a412ba8f9caaed8a596c532b83703dec072` |
| `bench_sm120_runtime` | `True` | `2271168` | `c929c0b2ce92a147183e9674129c4a053a68479f5e383d5831947c519d587eb0` |
| `train_gpt2cu` | `True` | `3119216` | `064e1e95dc83339da662d37fd658aefa7702bc48d1c5d6ab4e88895234a862e0` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
