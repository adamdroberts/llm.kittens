# SM120 Round Manifest

- run label: `codex_sm120_round_classifier_loss_only_x10_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_classifier_loss_only_x10_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_classifier_loss_only_x10_20260520`
- device arch: `SM120`
- max steps: `10`
- git commit: `0f21747`
- changed paths: `455`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2128288` | `6b8c36697fda8c582118775ac54f2e247bc7de112cf00b1eee534b5692e3da33` |
| `test_attention` | `True` | `1760032` | `35514c2e09673eaf398330d50e94f2702aaefbafb07d5e48a287ae67133dfbea` |
| `test_layernorm` | `True` | `1237784` | `80d8573bdcc7578d4ec561c83e140093bf1f93f8b2640980e66e8139149f34f1` |
| `test_bias` | `True` | `2039504` | `01710c4313a1b92cb3ce2933fc69988291356690eccef249fde072c4b4a8d1df` |
| `test_gelu` | `True` | `1139336` | `788f7c8d41cce6199ab39f2523a9ed5c9fdc92e44679120185f1f96223d33137` |
| `test_fused_classifier` | `True` | `1164032` | `e915d1e2fec51eee37d2d57fb5a4eee8cd50f5c56a93e2ad256a152c5ba5b13f` |
| `test_encoder` | `True` | `1165512` | `79e57bc4f97e0619d6f165926a9e2f01c7e033f9d89f5fd55951a94e67c12d68` |
| `test_adamw` | `True` | `1138408` | `59f73ed5203ad852d1a5a51e937a6e50315662dd4cadfc793ce2d2972ed36fb8` |
| `test_global_norm` | `True` | `1138816` | `2b466d791ac697124f92ca23f26104f169b4ec3d6af14d4aaddd7f2dbc31db38` |
| `bench_sm120_matmul` | `True` | `2365064` | `d71e8cce6d2d2c90658b678ed40d28a45de0d5a6b8a87ee630c3b12f68ebb846` |
| `bench_sm120_attention` | `True` | `1731352` | `0429b6526a5ddf0cd59577ff8a21a323473e3be79e98bf60acaca72e84b5b11a` |
| `bench_sm120_layernorm` | `True` | `1233728` | `79cbeae6a72f725d3c103386a974599e7e296d5187e89e64f6dafa1f2faf9346` |
| `bench_sm120_runtime` | `True` | `2186000` | `703c0822e63260f7bc33231c6a1c60068e664e38aaee956f636fa5f250f3fc5e` |
| `train_gpt2cu` | `True` | `3037016` | `fcf52e2483a2c3a3c5ce272bbbd0f03914347bb3634577eb519293e57672c9d8` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
