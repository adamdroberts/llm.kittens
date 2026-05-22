# SM120 Round Manifest

- run label: `codex_sm120_round_torch_stack_refresh_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_torch_stack_refresh_20260521`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `482`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `9c4b09a6ac51f1088fcab1b2a96d438e4413b6853b1317110d0da9ca39f0099d` |
| `test_attention` | `True` | `1760032` | `a6e995c6043d2dfe28093477df2439f92537c32229926de625570fb02cc60871` |
| `test_layernorm` | `True` | `1237784` | `ac9f2e916d15a8d59a6a4ff6925429f02270891deda4fdc06d68be60f952c9b5` |
| `test_bias` | `True` | `2048616` | `69144f2d4a49e7ed50e18546496c04cf73ef86f354fe6901214e1c1e106704b7` |
| `test_gelu` | `True` | `1139336` | `6d8a139dccf104983e40343beb56db66e5b3a9f77862f42878984b82ed722e60` |
| `test_fused_classifier` | `True` | `1164032` | `0d0f00165f2e362f7aa8aaa598d00ef6d59cde2c48f659064dd195f4a4bda840` |
| `test_encoder` | `True` | `1165512` | `6822272c627cbb6a4abbc7435c23c9a6419a599b5cd6dacc04378787586103ec` |
| `test_adamw` | `True` | `1138408` | `1915df9b47d1c85f35268328dc35a622d3fb6538fee58a4e5968cfe0a37b8b17` |
| `test_global_norm` | `True` | `1138880` | `9fd3fd5e08e841337bd08e171b56d39a4820b8fb06d920fb4c56944b62e0cad6` |
| `bench_sm120_matmul` | `True` | `2373912` | `307a767b2a726d6f25114d2e16f7d9a632c0b69fd68eb5a03225873697337a31` |
| `bench_sm120_attention` | `True` | `1731352` | `d78ddcf7f6e7ad745b99c40a31ba2a7c6f3cab568afe20428a3ad2419aaf7001` |
| `bench_sm120_layernorm` | `True` | `1233728` | `a340b65ed3fc27f261324581154764b276d3322593a7c34a1a5930032d1920b7` |
| `bench_sm120_runtime` | `True` | `2217576` | `8d3122dc84c5be96c787aba5cf76012b64c25d921b3a816186b96df6c7dc64b5` |
| `train_gpt2cu` | `True` | `3045944` | `e7d5c658fba8b9c86e638d40e149b0b756d30297fb9d9b84262305cd1b18b974` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
