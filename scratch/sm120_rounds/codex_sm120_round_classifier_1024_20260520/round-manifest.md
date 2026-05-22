# SM120 Round Manifest

- run label: `codex_sm120_round_classifier_1024_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_classifier_1024_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_classifier_1024_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `434`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2115600` | `813405542f76a70cce8efa73ed7a0b824818a0b153b073646d4e1fe7abc77217` |
| `test_attention` | `True` | `1760032` | `040c4e5c1e8bca2c51f5b60517a115710d771b9686433576e736a17b12c06f96` |
| `test_layernorm` | `True` | `1237784` | `002d7d3095493f7c0a34e1c9d5d57825ffead1e674e4b6f2562706180495283c` |
| `test_bias` | `True` | `2039504` | `f25059371bd8139b53b6f0eb4c400ba71170c4f63fa0a5553d0a2973c65c0d03` |
| `test_gelu` | `True` | `1139336` | `75cbc6db8216eff3d8d8af66f9671c5246531ea27357375d7489aea88158c039` |
| `test_fused_classifier` | `True` | `1146768` | `749d1eba556f637b9d34e59f2136a69fd09c9be35fba566f87421905870f95c1` |
| `test_encoder` | `True` | `1165512` | `4e2739dc48f4fdad99c68ec21b1b43849c9e252d0a62b8049a6497c1d3c0c95f` |
| `test_adamw` | `True` | `1134168` | `8031cf146d13e76f9934d8787e321b853392ef2af0008958d3a8b4bf385bd290` |
| `test_global_norm` | `True` | `1138816` | `ee7cae82160d9a967151efe04dd3f2603f046b69093d08af1f9c66772459fc6d` |
| `bench_sm120_matmul` | `True` | `2344376` | `92323a62617ed3758f99e687cf6471156e6f17acf302b18a2bb63ce9f1e7e0b4` |
| `bench_sm120_attention` | `True` | `1731352` | `7569057dc1c10c5f2d061be16ac79dab1894c53ccc8705d4427e39303f0cca85` |
| `bench_sm120_layernorm` | `True` | `1229088` | `7eaefcfbeab01920495d3da95a06ec6d837306003009d2b273cb538443a617e2` |
| `bench_sm120_runtime` | `True` | `2168552` | `5027fa8dee68ee1b23cad129422f44fbf75fcbbf6a481af74fe68f020b59aad2` |
| `train_gpt2cu` | `True` | `3036536` | `079f6b6c9156c99bbee818b853182ad80cfe0036b0033ea8f415b3fc02e14195` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
