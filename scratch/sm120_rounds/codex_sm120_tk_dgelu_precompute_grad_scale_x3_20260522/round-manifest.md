# SM120 Round Manifest

- run label: `codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `670`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2190792` | `ffb797d9f4080f23ab81b7e565789f004ad048b629b40415a5d4bfd5219f490e` |
| `test_attention` | `True` | `1800528` | `98075a6f3f3cc4fd3e65ea0ab3c225f027080a4737f8e0eab3d9597adc32bb55` |
| `test_layernorm` | `True` | `1278296` | `e70a0604cf21f4c8ef4143b908e3be27945773353c839d3236929176af301bc8` |
| `test_bias` | `True` | `2052256` | `3cbf6dd827b651af6c8677e835f82bb13db1cf2a4b289681686d821c700ff693` |
| `test_gelu` | `True` | `1179912` | `ca96b151c628f233c1f24ea54e6e77a330e96aedf56cd75e52dbb1e24722d217` |
| `test_fused_classifier` | `True` | `1208704` | `f65f4c5af647e2700e2b86e1c1e8479342b0bea23909f5dc85a09ac5c9c672f5` |
| `test_encoder` | `True` | `1210168` | `b2dc217a3c96fbf54f14a29c88532325051dfb982ec6596d5320392ccac87871` |
| `test_adamw` | `True` | `1183768` | `5e18216fc26ecf3fd9c136c8a1a04dced9e7ad8351650f04cd65f8615daa8e32` |
| `test_global_norm` | `True` | `1179464` | `3128d75b34014a26bf38c440b646dd447434cedf949f77a0a9560c12b7240f25` |
| `bench_sm120_matmul` | `True` | `2369344` | `860e65833c41b0dd2430bf02feab3e3ea4e6d133ee1decc87790e637c9d8bafc` |
| `bench_sm120_attention` | `True` | `1768800` | `0f191723845258d15d5b4c9892aafbb82b46fd1ba864552c889692a6958b54eb` |
| `bench_sm120_layernorm` | `True` | `1274232` | `c87d2f13e2422a6009ef139728ffcf563f14c9cbb39a192d58f5564b0350131a` |
| `bench_sm120_runtime` | `True` | `2234304` | `608a5e0662c5c36a5087250218d8c53f3b14d76433994bfe3885501da085a5a7` |
| `train_gpt2cu` | `True` | `3114128` | `e28bcb3ad4a65f8da49112196f17c7a34c354574825e1884cd424b8923e1396a` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
