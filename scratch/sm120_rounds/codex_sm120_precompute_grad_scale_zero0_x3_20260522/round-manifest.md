# SM120 Round Manifest

- run label: `codex_sm120_precompute_grad_scale_zero0_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_precompute_grad_scale_zero0_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_precompute_grad_scale_zero0_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `654`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `1e21758d739056af7edcf700665d078412a9bbbfb64fc8f83ef757bb10bbb550` |
| `test_attention` | `True` | `1800528` | `28a447d7c29b1c3fd144ec9553880cfe5d68dc931e399ed125e3d729df14a502` |
| `test_layernorm` | `True` | `1278296` | `af3e70f936a7d9ebff5dee4fa8f3dc96c67ea322fdcffbe5e0510c9403e0de22` |
| `test_bias` | `True` | `2089120` | `c4811cd8f88adae349e982f66aaa75df4381256fbd34e406bf90869b2565e2a9` |
| `test_gelu` | `True` | `1179912` | `b014518b694d2ac055680db7999400ac815593fd2172ad9ed4263bcd9745c19f` |
| `test_fused_classifier` | `True` | `1208704` | `d1bb294e5909139679340e57967c375c35473a71764f76ddca6e63bb1a709590` |
| `test_encoder` | `True` | `1210168` | `4dae534d846582087a602f064129e71b222d3e549d837cb3c841e97110b37b2e` |
| `test_adamw` | `True` | `1183768` | `c5197a60a85e02c526b3f30e7c950cc91e38d580f0e53c571ff3f1dc67ab027b` |
| `test_global_norm` | `True` | `1179464` | `bef3f043f2a6dc014d3fa08ec4489a2408008abf73307e64e08e641b73f282d1` |
| `bench_sm120_matmul` | `True` | `2410304` | `bb571f1e9ce63dbb76b4310ccaee0cfef58d5fdcdeb8b88486db80d3310a9730` |
| `bench_sm120_attention` | `True` | `1768800` | `b52e90d79ffe6aec2829b75f0bbfc2655d4f4b9dea07b13c24862bdca9cd9397` |
| `bench_sm120_layernorm` | `True` | `1274232` | `1255598ddbe2bd23ac0c6a75411750afdb5c5aada9984173f69357313f6af037` |
| `bench_sm120_runtime` | `True` | `2271168` | `026b20ff00f9a1e31076b6e462b1679accc9846ecc6455e18241b18aa6f1f4ce` |
| `train_gpt2cu` | `True` | `3119216` | `04c4035340ea643f2afc3995dc17d01f9dc8015d441fdaaef1f1d162e4edd5e4` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
