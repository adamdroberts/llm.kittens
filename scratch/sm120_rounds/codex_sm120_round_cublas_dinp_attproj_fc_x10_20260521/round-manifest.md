# SM120 Round Manifest

- run label: `codex_sm120_round_cublas_dinp_attproj_fc_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_cublas_dinp_attproj_fc_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- git commit: `0f21747`
- changed paths: `481`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2141112` | `0728d3fedfbb7cbd1235ebbc703175f39e02a3b92c1e9c65114516306c0d9f19` |
| `test_attention` | `True` | `1760032` | `ffff8ac66ed9cb7a89d712ac619461db4c825944e737a0c05d4d13a107a5a1ef` |
| `test_layernorm` | `True` | `1237784` | `a4bbd99152b95eb16c95c3ff3cd8f94d58d52b24d6f571c37bb2128215958068` |
| `test_bias` | `True` | `2048616` | `98599c52bef0962997a0a8ead0667326337bdf18c5204366c5d87c462d661b73` |
| `test_gelu` | `True` | `1139336` | `c8a4da5e17f78a7efe499ffa2a8989ee925933c55441a7781278a08ca4cb414d` |
| `test_fused_classifier` | `True` | `1164032` | `5f320c91e133e99c48a2262ed2d65350a3bc4f0c66061aeca819f7e237b517d5` |
| `test_encoder` | `True` | `1165512` | `3019d20f9aa707d228530f7c97d1630a9ed925e5e23c2be994f91f6081863046` |
| `test_adamw` | `True` | `1138408` | `b666b9f93d371ca88fdba63f6e65b97fd98e6fda888017c0a667feee5624a67c` |
| `test_global_norm` | `True` | `1138880` | `b1726f61b2ae51b4a3b59154fef3a5c73659c521b7c280b229417de375013fad` |
| `bench_sm120_matmul` | `True` | `2373912` | `7a2c786e2e730a54fca0f5eb01d37c8497f1d656d80d96cdbaf53c3628797931` |
| `bench_sm120_attention` | `True` | `1731352` | `8d8cd824440af0a0f581dadcf2f218dc3b072bd50ee88e6f342020ab433018e8` |
| `bench_sm120_layernorm` | `True` | `1233728` | `5d49ca50853a7baee06cbde83c70c082f323b10c3575570a8dcf2237106d5c02` |
| `bench_sm120_runtime` | `True` | `2217576` | `54878982b8fd54a80eb1df62b147b7d0c171c2c738c7e2de869badb14cbc86a6` |
| `train_gpt2cu` | `True` | `3045944` | `f334ed31e49c3386d27671a3a1f6625eb10121525ff2c4f027b749a0b4eb9620` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
