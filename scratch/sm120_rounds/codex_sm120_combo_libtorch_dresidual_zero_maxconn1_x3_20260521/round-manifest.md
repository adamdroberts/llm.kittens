# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_dresidual_zero_maxconn1_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_maxconn1_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_maxconn1_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `571`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `619ad549184bdd5d4f4f4c3446e348f78dee7ba7d29c2ea228c3f0946436809c` |
| `test_attention` | `True` | `1800528` | `d7d48c71e50dd14f0af43d9ed2bb8f4358d4403162cfd5c64a968d98f9f70722` |
| `test_layernorm` | `True` | `1278296` | `a74ee254b6f5150ea150930a8a74127b595a5a224af48271f8d10f664075b14d` |
| `test_bias` | `True` | `2089120` | `52bf4b1b9dfe464c952c0da8255520c4e3184eff546128ce43eda29662bc6f52` |
| `test_gelu` | `True` | `1179912` | `eddefe581746c5b91df3dd0db7c5e2594af7cf0014233d357fcac9bc7c6016b1` |
| `test_fused_classifier` | `True` | `1208704` | `1ad8a18358992d18483e242be072d64c751979d93b954daee57443e6bd961173` |
| `test_encoder` | `True` | `1210168` | `893f906bf31f95c29597c208d5d25b5598bcdb4043eaa2487c2103e629080974` |
| `test_adamw` | `True` | `1183768` | `13519e1d271b0a6859084393e4721a9a9c18262be45662010d6f94e475fc8c42` |
| `test_global_norm` | `True` | `1179464` | `3ba782b8509237ed303e6cead777a9be796edfde3abcf5937618d60ee05c8dbc` |
| `bench_sm120_matmul` | `True` | `2410304` | `15900802d94a096bd814771f5ce51561ab25a1f0e39e923167eb8e8d1a2b2d68` |
| `bench_sm120_attention` | `True` | `1768800` | `3bdd1b77a335aa6992983130f17987bd2917fba39af23a32cee22b3443232015` |
| `bench_sm120_layernorm` | `True` | `1274232` | `1abf79713e1e61f3f1915c49b60c4fb7100d709d7acb361a0ac73dc8a87dfc0f` |
| `bench_sm120_runtime` | `True` | `2271168` | `f204b41d2a77b214df6afd7503b91d6f9e376d6917cb3bedec371ad841c15882` |
| `train_gpt2cu` | `True` | `3105552` | `552c690c61c538d07c011a054085b51b2f3d764027c74297893d7c642d3a3615` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
