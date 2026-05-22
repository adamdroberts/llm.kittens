# SM120 Round Manifest

- run label: `codex_sm120_round_backward_stream_sync_default_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_backward_stream_sync_default_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- git commit: `0f21747`
- changed paths: `491`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `b19b7593bea13aa161dea153bfc96e0ae734ed1797554a098ca0e5b9753b276a` |
| `test_attention` | `True` | `1760032` | `c4247d4a9b0dfeb3dc37cd232e9a83995d4a1827e1718834edde1fcad3c2dc9f` |
| `test_layernorm` | `True` | `1237784` | `4f816035969538a07dd12573bf044a4dc0ab4daf6cbcdb5d836d1863436f7834` |
| `test_bias` | `True` | `2048616` | `7b98957f0287dc2ebdad19dee7afcd3e94eb233c91bc32f4f4a6c4ef43ce5094` |
| `test_gelu` | `True` | `1139336` | `82c8952b2896d5a800f5a8c062e34eb11a17c68cd52a91021f19c70559df9ef6` |
| `test_fused_classifier` | `True` | `1164032` | `3c9c698debc8ba688801c367704f83443ce6d9e0ab09002dac33901b388a4528` |
| `test_encoder` | `True` | `1165512` | `0815645e2dafd967d9783159205134e644c4232fa86283d0469115af770d2ed7` |
| `test_adamw` | `True` | `1138408` | `ad2bfbd7fece6d8176ea677bfb062315f58f7188e510a33cf070d48397227a3a` |
| `test_global_norm` | `True` | `1138880` | `83c28aa050695cdeb5b90c6bd563922ffc57063cbdbc22ec85e078d5280e696b` |
| `bench_sm120_matmul` | `True` | `2373912` | `98fe6f12a6461d75898df4f72609d6cde9f5b84354c5fbfc7dfdc3af58f1b295` |
| `bench_sm120_attention` | `True` | `1728312` | `77d975d69f4c1bad815b79040834712adf14815e53d1d311ece851032396a068` |
| `bench_sm120_layernorm` | `True` | `1233728` | `93fb8f3d7ba7790569461985c7bebade3a813cc818b054333da68ad811d091b2` |
| `bench_sm120_runtime` | `True` | `2217576` | `8e412d403a57a6708c84d40d1191c131b9e3b51236749c3f64a1780842cba4a6` |
| `train_gpt2cu` | `True` | `3045944` | `b398f6cc7e7d8ec47e6de154b6e8f6917ed9d3257079f72b2c39bdb1f059c344` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
