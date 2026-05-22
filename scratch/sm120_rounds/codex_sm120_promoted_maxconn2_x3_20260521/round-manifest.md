# SM120 Round Manifest

- run label: `codex_sm120_promoted_maxconn2_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_maxconn2_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_maxconn2_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `609`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `f30ea229efa27e9bb265b5f7673f87162e2661e03ab8b5d348ebe6b5f7bfd839` |
| `test_attention` | `True` | `1800528` | `213acd406b6bbca5d0119d14b49823519d0fe22700b2113ad9911ee1c39f3ff1` |
| `test_layernorm` | `True` | `1278296` | `8f67b816ce059d7e4a58bc637981c12995c63e84b9182971306b1c7897dd0162` |
| `test_bias` | `True` | `2089120` | `fcc39e4806ddb5bd3497192d5f75c6ffc7134fee31ca7293986e28baae8a39b4` |
| `test_gelu` | `True` | `1179912` | `6bca326a690396175938db5e1fdaf9c3741d71d94ad627ba72648ea4d8a944df` |
| `test_fused_classifier` | `True` | `1208704` | `76d71bc20771b0c8138c468ca958e34fde490ed7e6698bbc4c83e73930e79a35` |
| `test_encoder` | `True` | `1210168` | `709ee48cdb7aeca812fb592b6ce0ee2630c0ed166d39b6736ba37aa1925b8e87` |
| `test_adamw` | `True` | `1183768` | `ffe72c135a7233669fc48a544722ee12cec88bdfbd82fc930fd047af67aff27c` |
| `test_global_norm` | `True` | `1179464` | `42c3aaebc7e147c24b1f5203a58a8e9a814238772e13abe304c461892273b886` |
| `bench_sm120_matmul` | `True` | `2410304` | `8a21694ac9242de9bdbe1bb7988623c2f0500d19d1e2406f05468778d221ac12` |
| `bench_sm120_attention` | `True` | `1768800` | `5d06d358496e849fda3cb73f3706a0487af5862e4faf4172bd5817bc9fc2435a` |
| `bench_sm120_layernorm` | `True` | `1274232` | `4375d3de3d92c33217f7edf80fabaff40a41303f433d533ba0e2fd227a59f418` |
| `bench_sm120_runtime` | `True` | `2271168` | `7689e0a5cd35c32989ef0de6c5fe43bcfc4e3eba93bf70e310ce2f902d2a94e6` |
| `train_gpt2cu` | `True` | `3105552` | `c9ecff72fb44849fe20bee7581361f04076e7464b02e0b0aadc3b3c04d6adc5d` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
