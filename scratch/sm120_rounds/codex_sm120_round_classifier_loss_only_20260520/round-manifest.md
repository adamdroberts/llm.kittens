# SM120 Round Manifest

- run label: `codex_sm120_round_classifier_loss_only_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_classifier_loss_only_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_classifier_loss_only_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `454`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2128288` | `7f6f57a32b30470dbd77e7a477754da0bcfe4eeb06fd76767b7d5534cda8538e` |
| `test_attention` | `True` | `1760032` | `5a905a9c4da6eaba34b6a441b66ad5d55a034af280418cb0b0587901925b62a8` |
| `test_layernorm` | `True` | `1237784` | `1aa6dea74a2c85192eb78c55418afcd165571c2be3ea24e51fef22d712c7a7c6` |
| `test_bias` | `True` | `2039504` | `0968d3d0aa4e0966f4b2820b65fe63072de1006326e5d51577a7f3fc0d7233c1` |
| `test_gelu` | `True` | `1139336` | `d682b4dfcd4294c5d3930c20e4f1a50018582c02932083b5ed062c7f8586590e` |
| `test_fused_classifier` | `True` | `1164032` | `ce96bcfbb06210630f83ea3b8bcc163e2b72e59c58dfba9c99921150bed14fb5` |
| `test_encoder` | `True` | `1165512` | `59e45899f7b6c19fdf0dcfaf55844636aea6a0f67a5915b4a15fecac8d3c1303` |
| `test_adamw` | `True` | `1138408` | `e4655214c991abe1962bcc65d6b1d7e6a9644bd1157b7067fe21042552f2c686` |
| `test_global_norm` | `True` | `1138816` | `b927efd977456f9d0cf2a61e3f1efab2f3907a1f947d614daf1039fffd7a625a` |
| `bench_sm120_matmul` | `True` | `2365064` | `08a09f9b5afbe3768664e131b75f1c2f8476c3c328b8c28554db27268bc0ac4c` |
| `bench_sm120_attention` | `True` | `1731352` | `f7b206479c50053467a84a14fd88c8d64dc47ff20958e4462557af20f9aba52b` |
| `bench_sm120_layernorm` | `True` | `1233728` | `ad8bd4dc7ecaf08f61cddfe26a7aaa9f45ab6dcb1bb31058cc32e3ab3b15b71c` |
| `bench_sm120_runtime` | `True` | `2186000` | `59a5c485fa75acf599a676fd04d7ebdaab2f1f6d8f084c0d18b8359625d25d67` |
| `train_gpt2cu` | `True` | `3037016` | `ee3fcef800fa9d65ca5774005859acf2c33e369a1c81f32b7c5f0a5a519ffb52` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
