# SM120 Round Manifest

- run label: `codex_sm120_promoted_maxconn16_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_maxconn16_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_maxconn16_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `635`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `dcc8c59b6c73332f3d843b022a64485faa5b7596152c6625ba401ad839297b17` |
| `test_attention` | `True` | `1800528` | `beb723d5741b54f323a3b438012a2d8cfd181ba2cc6d42337ef77c6156255d60` |
| `test_layernorm` | `True` | `1278296` | `75162e205d79b8605959a62196be423b3a00ef938fa5078667407be8758857e1` |
| `test_bias` | `True` | `2089120` | `fe88f2e279a181941424daf745d4defc08d61a217ac9af0dc3dfbe47c61dee3e` |
| `test_gelu` | `True` | `1179912` | `cd58a95b3cd628fc5b8a10a39eb30e643c9179f4448e6e230326fc2c906a4a4e` |
| `test_fused_classifier` | `True` | `1208704` | `76939c422b79b003485ef9968a277305c1c35a5329f51585740b84bf50f0376c` |
| `test_encoder` | `True` | `1210168` | `b2377119b8824644f686949edfddd009f135bc6d3ff6ba7ea2212f2b1d6f5e3c` |
| `test_adamw` | `True` | `1183768` | `6d8c01e091d1d48bb0785cc6643110f4696b14a1e6c2f9e6ff96c2bbb549a9da` |
| `test_global_norm` | `True` | `1179464` | `af897d2008e39965e258908bb5a2f3484d8196470836b38838f91fbc33439ef1` |
| `bench_sm120_matmul` | `True` | `2410304` | `480582920a29bfa9dbb15ab605b555de8ff50c0b37f62d773e50b362bec1e944` |
| `bench_sm120_attention` | `True` | `1768800` | `6cccc4b273f47315eeb208291176ce31c4922efc4381c6eae155983b098d044b` |
| `bench_sm120_layernorm` | `True` | `1274232` | `99cbe5fe704217bff3fcbc34e599d83417bcd62325875c8beb26750d8807b010` |
| `bench_sm120_runtime` | `True` | `2271168` | `85af4b9d0b37776ded58302759af0edeb41b21a13452d5090d32c4ca1ac3dbd1` |
| `train_gpt2cu` | `True` | `3105552` | `83bc22c45f86b02d2eded37a23458eef6950a6b9b35074e6ae5a54f9f4779aed` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
