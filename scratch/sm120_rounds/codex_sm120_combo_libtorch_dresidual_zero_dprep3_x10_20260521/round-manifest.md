# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_dresidual_zero_dprep3_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_dprep3_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_dprep3_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `589`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `0f8cd55e16018db41da6e1132d283a8a67d3aec55d49a7628a1164a67a548493` |
| `test_attention` | `True` | `1800528` | `3c21346acf3d99557ec2c296c976bd6e80fe81d7554b6785a91c1c5f7f2f9a49` |
| `test_layernorm` | `True` | `1278296` | `a5988d16c346167ad3c3119354b25e99bbe0b1efc671a5a4bc243b30dfa37c2c` |
| `test_bias` | `True` | `2089120` | `9374cf933254175764f13ae0f6e513dec5580c65056553f69b68e0847e1790bf` |
| `test_gelu` | `True` | `1179912` | `8a369d986f603858fdfffe1d755b61eb7a809073893e19f6cc3de2f3fe8bfa63` |
| `test_fused_classifier` | `True` | `1208704` | `0a27def4802399ee78011adefc1f5729c97f4e4df1e20a59c4271a912e20308e` |
| `test_encoder` | `True` | `1210168` | `90a4a0784da18bc6f89a4cbeba18d41584edcf2400c2b334fdef27a62d859337` |
| `test_adamw` | `True` | `1183768` | `9a437e1844994cc635c22847981d8cdb2c63eb78965091032be3749123ecb189` |
| `test_global_norm` | `True` | `1179464` | `a91afa16720214604aed7cb3544c1979cc5cd646c358143422a82060d61e4db0` |
| `bench_sm120_matmul` | `True` | `2410304` | `7f7f79f387150d726d4a05c76708d6377386b01fdcfd8b4c4473494acebea8ad` |
| `bench_sm120_attention` | `True` | `1768800` | `6eab3e51bf98930746561582add8b4b059efff7269ff2ac09ae8347d7f79317b` |
| `bench_sm120_layernorm` | `True` | `1274232` | `660f664feffb0974d994a45764676f4788a5233dcf200f796b16d0b8233aee72` |
| `bench_sm120_runtime` | `True` | `2271168` | `d7b9e4aa3c651a6714abc40833e8f6ef2afed2329c381d7563d14f96dd5de052` |
| `train_gpt2cu` | `True` | `3105552` | `da88989d606f570f3ac2e6791124bb0c75e86244bca78a8b101ade68fc6d7a81` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
