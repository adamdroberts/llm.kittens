# SM120 Round Manifest

- run label: `codex_sm120_global_norm_block768_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_global_norm_block768_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_global_norm_block768_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `679`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `23f52b8133c685ad4769f8a9d2e391938fa7daca2f5279320db071023209d7a6` |
| `test_attention` | `True` | `1800528` | `eb691b0e546b30468081b86d87004f3775d1f9d091da9f0b4ccbbf812a51e9af` |
| `test_layernorm` | `True` | `1278296` | `25f20b13d124beab4a7ef53904e7f16e405097e5d04b2eee7d859e346874e883` |
| `test_bias` | `True` | `2089120` | `8e4b1ee0b07ea0c4f055ef7a2b18522e7f5b4d7f28843d4d5661728e844c95b5` |
| `test_gelu` | `True` | `1179912` | `b398313c8212f6c59d375820e21dee9a134e96b4dbdb18d99f97ac518a0c07c2` |
| `test_fused_classifier` | `True` | `1208704` | `1361ffca3dd6fbb974c3f2989b1fa5b8ea3104594ad211630a495ce8c8927de5` |
| `test_encoder` | `True` | `1210168` | `42cef5929f37c58e3c2e2f81f3552da647425aa8b6902e0da08194907c383929` |
| `test_adamw` | `True` | `1183768` | `717484f5999cd0b7ac1a31f11e58df4bc0ee0f43c7d04b985cbb5678943614db` |
| `test_global_norm` | `True` | `1179464` | `421ab90113e0984ce1422a86d23a768520adba3f9f0c9b34445f64c6a17eeed6` |
| `bench_sm120_matmul` | `True` | `2410304` | `0cf011c4542c8c17069e20402d83f71127db2c21925f6ca8f06d9f659c6cbd6b` |
| `bench_sm120_attention` | `True` | `1768800` | `f95fe0670e130a629f1b60712864503d27b76887353e131a8637d569c8c8ae0b` |
| `bench_sm120_layernorm` | `True` | `1274232` | `6c0676af4392484e7f36d4b07f7fc14459dad3cfb8e3814406fa75ef325809b3` |
| `bench_sm120_runtime` | `True` | `2271168` | `0861bbe45253a94504e3a544d144cb929ed4caad29c3a1f506cfb0de9d649158` |
| `train_gpt2cu` | `True` | `3105552` | `49efaa572bb185af50706b627f0e2ff37bb558c3c61a634b821d700141eb0a5f` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
