# SM120 Round Manifest

- run label: `codex_fresh_sm120_round_20260520_gpu`
- artifact dir: `scratch/sm120_rounds/codex_fresh_sm120_round_20260520_gpu`
- train output dir: `log124M/5090_S_codex_fresh_sm120_round_20260520_gpu`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `430`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2115600` | `2a5c5f74fe1f0c202b1767d459cb25a2f33da33570a747d7fbe2ffa262730f51` |
| `test_attention` | `True` | `1760032` | `4098edc48f31cb9663383453261d0c7ca056bc92dc9d840aae4b51c1d5a7384f` |
| `test_layernorm` | `True` | `1241880` | `1e47219df7c84b874acbb2362e9e4e2c622b3a1b04e5ba28488ca94ef4e5a1d2` |
| `test_bias` | `True` | `2039504` | `59abde30ca3b9981d70093f988264b6c59a3c38128580292926deb860bb90335` |
| `test_gelu` | `True` | `1139336` | `c14ef94d7cd768f7ddd05395f4b0131058e0fa8d6d8d649c26ae0fc00a62bc37` |
| `test_fused_classifier` | `True` | `1146768` | `cf5271da9bcb1aa7f1ee47305244e842fb28bad27813c5ce94fb1490aa424d0d` |
| `test_encoder` | `True` | `1165512` | `028aa6885112e445266d03a7e486c8ee24e5c78e25d8653b3bbace0fa334e01a` |
| `test_adamw` | `True` | `1134168` | `e1c80867cda4900eb70198925c369175d5bf990f095dd917dad9dd6c9c3e5486` |
| `test_global_norm` | `True` | `1138816` | `4c0526877090e12925aef15fa1c50b07bacf34d53cd946249aca95103d485658` |
| `bench_sm120_matmul` | `True` | `2344376` | `b0c2a4a2e6f34f2e53d805d6620a1b529259f7a5af212c8466d6cce9e43be71a` |
| `bench_sm120_attention` | `True` | `1731352` | `cb845813c31db00a1a3c0b8b808c5daf8e0b8762f122b8fa76a64c073056390d` |
| `bench_sm120_layernorm` | `True` | `1233184` | `bdd24765bff3f7bb003c0ba8f07d480d21a0260adf0373869f13dd9eef3910c6` |
| `bench_sm120_runtime` | `True` | `2168552` | `f104eb470c20f4bcf19142362d8892d191842159d2098bb3cb17de36247dde02` |
| `train_gpt2cu` | `True` | `3040632` | `db4ad7b6a208ad4ab7cae246bebd128160718624a6b5de0c7706f618195a0688` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
