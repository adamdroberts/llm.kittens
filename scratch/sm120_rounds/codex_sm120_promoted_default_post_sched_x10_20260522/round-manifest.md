# SM120 Round Manifest

- run label: `codex_sm120_promoted_default_post_sched_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_default_post_sched_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_default_post_sched_x10_20260522`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `637`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `81dcfe00c0b85b6a9adf040e8f33b244ac9d56cb62914d0265452b9c4c2ef9cc` |
| `test_attention` | `True` | `1800528` | `66ec9dbca6955585a9e19ed6a8f2b3e5892b1a8cd75fced3aadba425a55ebc8f` |
| `test_layernorm` | `True` | `1278296` | `efeea936d9196a62132212e7dd0b2185240e8df363f784f809a7a04029a37018` |
| `test_bias` | `True` | `2089120` | `a06d501af556486495e1b50ee2ecd1a0d612fbe1c04d88d5d47f4338f32df55f` |
| `test_gelu` | `True` | `1179912` | `e376528f7365209b3df0ad0d0005ed61e5469303c753ac769fcac94ab012a333` |
| `test_fused_classifier` | `True` | `1208704` | `03dd783d6c4b808db1934bc6b5562818725a829278deaf39a07a8042c4a78c82` |
| `test_encoder` | `True` | `1210168` | `230d489d50e66c1b4561df8243972192b96aa564070f82b3ad7dd72ebbe33fce` |
| `test_adamw` | `True` | `1183768` | `283f79f174c9cff57dbf65bae4cc688a42f1d306ccc4bd0564731b93d7e70390` |
| `test_global_norm` | `True` | `1179464` | `455f8ec672af905620d2de8b5ec4d9bb8fde499b79e9dfb2ea79f21c7e659282` |
| `bench_sm120_matmul` | `True` | `2410304` | `172fbcf5b43d664f10ea5627e11637d2a14e0b9d924bf0494bbb2641d04084cb` |
| `bench_sm120_attention` | `True` | `1768800` | `e2779de7baa8a0438ce7e35db7bdbe7efa678dfac58e5ee9a124610a148e682b` |
| `bench_sm120_layernorm` | `True` | `1274232` | `35bb9737f16f44c8c111e439c3e6a12bf2a894532b6f999777c2f00533631928` |
| `bench_sm120_runtime` | `True` | `2271168` | `d54425051837f0fe3d44013b02cc33690ec3336adb628bb8b429f0419c89fb16` |
| `train_gpt2cu` | `True` | `3105552` | `d1db5331be49543b6a7388c86d95f99976d1591d5798a9885ee053235c0cc858` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
