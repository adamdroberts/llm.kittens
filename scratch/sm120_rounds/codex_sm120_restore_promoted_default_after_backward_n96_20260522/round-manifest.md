# SM120 Round Manifest

- run label: `codex_sm120_restore_promoted_default_after_backward_n96_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_backward_n96_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_promoted_default_after_backward_n96_20260522`
- device arch: `SM120`
- max steps: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `653`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `09266d127374de88a5276103a8fb355b6a9026c0a1dc8981acf1424f8a21ba0c` |
| `test_attention` | `True` | `1800528` | `2d6ab9715d85810ce0357295409a0ce6b9376ffa8cb55fc3a49e81f581538e31` |
| `test_layernorm` | `True` | `1278296` | `041856352921beaeca6693b82eacbe80b9801a6c2bbed21e25a62b3a1c082efc` |
| `test_bias` | `True` | `2089120` | `3f25e0e66ef93ba3a4da1d6aade071fed23cd3860e323c1299726a2438504148` |
| `test_gelu` | `True` | `1179912` | `0c753f1acf76384d11a422d8277dc84d7e3cf3df7bc49826330bf390670ea2d5` |
| `test_fused_classifier` | `True` | `1208704` | `91b52e1796991a76e68d9b5dfe100b9f0549c85403022e4a414665cb9e2616e7` |
| `test_encoder` | `True` | `1210168` | `429950ae1e64fffdec564c47c53a26f18f61f5bfa5ec30b92bb17702cf4006d5` |
| `test_adamw` | `True` | `1183768` | `985f688fcd99c35d2441b3d00bc158d9003a5f7cffad181b321ab42db6586a86` |
| `test_global_norm` | `True` | `1179464` | `32f9df29723ab6587b36d4c99019b973ac25ca97ac473462da1afdecd5d9f0ac` |
| `bench_sm120_matmul` | `True` | `2410304` | `9f27068ddcf69366737baab1a3c5b26bfb7f1c749b31f94467c2e2bbdaf26c81` |
| `bench_sm120_attention` | `True` | `1768800` | `2f5dc51d2d48d0c68d23abccf875cf70f7eaafcbc613df50f30697491308cd35` |
| `bench_sm120_layernorm` | `True` | `1274232` | `f0817fe54486e86a32a5463501c9782e3f5699cefc218751fbea318a4df295ec` |
| `bench_sm120_runtime` | `True` | `2271168` | `f07fc3a22c00c4f67aa0b97e7de2724b03f06a727b8f3449157821c0da708dbc` |
| `train_gpt2cu` | `True` | `3105552` | `79bc5e2e135e7b9710e7140a98aee4b9640990c29f457031f5c6ff0323dfa6d3` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
