# SM120 Round Manifest

- run label: `codex_sm120_combo_all_native_winners_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_all_native_winners_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_all_native_winners_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `564`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2154392` | `73ca2a2c4032a46eb97f30aa6108bd014ade6ae20852e2fb7865f297178ed5d9` |
| `test_attention` | `True` | `1760032` | `0abe1bf1f7639c46dde7099f44c2d61829cf6709b40265f467234c45bd62f1ba` |
| `test_layernorm` | `True` | `1237784` | `e30b57e51fae6731e133e08fc94641275411deef43d45f8b4a709b50dc57c0d7` |
| `test_bias` | `True` | `2007656` | `1fb10783ba3789f4864031141c35d9217c08d02b49168a05d8a961709c86185b` |
| `test_gelu` | `True` | `1139336` | `4b7c69b9c5b6278ec679b8b5a979cd3cbbf6577f45504a4a039d7f7d2c5c3c5a` |
| `test_fused_classifier` | `True` | `1164032` | `f45c31de8cc96d6cf00ba23fc4a0b6c4e3796ed4d53af0760b81d2ea7cc1e79a` |
| `test_encoder` | `True` | `1165512` | `51a3997a0854ded229026ae67ec4adebb03469aa7bed1c940fa2a70e0788d1ad` |
| `test_adamw` | `True` | `1143192` | `52898d692111ac94bc01286d6223dc5652dae082c5fbe6676d9d4f73b9dc92b0` |
| `test_global_norm` | `True` | `1138880` | `60ef903ffd360b5183ee460f906b25e4fc1c0fb749baa6f96ce8401e9c6cfe83` |
| `bench_sm120_matmul` | `True` | `2332952` | `ba48c2fb92ceff8435c22137ca852b91959d93277862c9aec3f426946e2a5ffa` |
| `bench_sm120_attention` | `True` | `1728312` | `1378e79c8c06dc0589fbcc59a384e8f08313d3700e67a07a9f578af40a8ee02f` |
| `bench_sm120_layernorm` | `True` | `1233728` | `37e3269c915371b3803556ef5a235164144a5054c6b4779b639210957e1c2a07` |
| `bench_sm120_runtime` | `True` | `2189672` | `1f48a5dfc3882916d5260905600b122bde43c7630beb90121e6b6511f291a12f` |
| `train_gpt2cu` | `True` | `3062760` | `3ff7d256457cf0f81d9d6ccd77b29a4f326db04833d3432c5d959e55e5ee710d` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
