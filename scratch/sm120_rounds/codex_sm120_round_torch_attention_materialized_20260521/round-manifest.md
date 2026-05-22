# SM120 Round Manifest

- run label: `codex_sm120_round_torch_attention_materialized_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_torch_attention_materialized_20260521`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `487`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `367c6dec37cdae1e629c6e190f6bdc37684c5beba29eac0f855ba9101c56902a` |
| `test_attention` | `True` | `1760032` | `bb8af70fe474b543ea92f159e1de9b5a5146d8b0deef1edac818695f5fe0667b` |
| `test_layernorm` | `True` | `1237784` | `34350d429b05a0486f765394a2a2d1aa3fc95b41e438934cca04fb5500de7994` |
| `test_bias` | `True` | `2048616` | `f3a3805e4f627c886abb35aff4d430138c62c50730797f50be95a8801228c6ed` |
| `test_gelu` | `True` | `1139336` | `588ba8a315a49abbe3c439ee9680e71407f661ee68d5bfb9c264920667c7096f` |
| `test_fused_classifier` | `True` | `1164032` | `82fbfa0e1926269888cc923cf67e6fdc902c4468e2282fa8e018173b6c83a163` |
| `test_encoder` | `True` | `1165512` | `7d636f8bc860619d8efb5abbb6ba9aefdce2a055a63173daecd27b03accab2ca` |
| `test_adamw` | `True` | `1138408` | `a790541a97a3159d6c26521bec2c02f062dbf5051f0be26aad97861cd0d5fa5b` |
| `test_global_norm` | `True` | `1138880` | `707577b85e0a251e68842cf8b6a990e6ae2311be5756d20f0e41d31af3f1f633` |
| `bench_sm120_matmul` | `True` | `2373912` | `33affb95b160196da099779a1c533ff5188548e204736144db244671a2761e78` |
| `bench_sm120_attention` | `True` | `1731352` | `f2d21d9d5764a0df23f4cb0a8d72e2332ed599ee1ba0e8a36c8690e47875b921` |
| `bench_sm120_layernorm` | `True` | `1233728` | `3c387c0097ae289c58b0aa373b97093260aa98ca55268f5ee091dc9e33ccff04` |
| `bench_sm120_runtime` | `True` | `2217576` | `a3c75ef9f0c90e66c8348ca5db3f675ba00bcdb0df69a0371aea55025502b489` |
| `train_gpt2cu` | `True` | `3045944` | `ee32f41bec937c6737e5ce36187e6bbab2a43528b39db5f22e5675b20467007a` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
