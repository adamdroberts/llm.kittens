# SM120 Round Manifest

- run label: `codex_sm120_restore_promoted_default_after_disable_cublas_bwd_maxconn8_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_disable_cublas_bwd_maxconn8_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_promoted_default_after_disable_cublas_bwd_maxconn8_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `644`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `a5c45a26ca3cbbbb4fc9316dec250d433ae80e7d13695edf84b69a8bdf103d8c` |
| `test_attention` | `True` | `1800528` | `fdccee961c49a08121b5338b91e18a9e29d5d74a7ebc250bc81cedf086cc6a83` |
| `test_layernorm` | `True` | `1278296` | `358ff63d0c23d35ea41a02874067fc33bef5104e9c9488eae3ff4ed90929d16a` |
| `test_bias` | `True` | `2089120` | `df2a8a97e96f3c453c6177b578a299e8d43fd66148783a849b9285f5386a67c1` |
| `test_gelu` | `True` | `1179912` | `1e92cd0b8a9208a693a8b5403dd21cabc72f00ae96730839ab6bd4f2694ed8c3` |
| `test_fused_classifier` | `True` | `1208704` | `48e085ac8ae79c4366d17442a1c7a5b92c2c45c8b9c80c9b7df57ca4a36cbd11` |
| `test_encoder` | `True` | `1210168` | `6e4d58923c692ee92d7a48380673c89fe820ca9c0f0ac1b3b5bd9fe1f0b403ca` |
| `test_adamw` | `True` | `1183768` | `7702c0f7cbaefb4542b9bfb51223e36a1ea1bd802caf37bacbe93ae944d43418` |
| `test_global_norm` | `True` | `1179464` | `2bed852b6547ee19f33962908b1f2337038aa9f779de85074e1420e3660d3a31` |
| `bench_sm120_matmul` | `True` | `2410304` | `44e6e19c6ba50f5d745f5c69990f0266270d62be79bf2c706f16402afb782d33` |
| `bench_sm120_attention` | `True` | `1768800` | `68672cf82cce862b722366e2c675ad02b1685bb964e208d858d8064b87bf74fa` |
| `bench_sm120_layernorm` | `True` | `1274232` | `7361d604eaeb9fff9d83acae4e5c3c058dfdc6e6e4a4205bfccbf0889386288a` |
| `bench_sm120_runtime` | `True` | `2271168` | `aba61efdc3a24b2074d25177b38ee73257c123c7b2ffabe3149de08b589e56c6` |
| `train_gpt2cu` | `True` | `3105552` | `0d638c6cd459e201ef06a0f8854c3d51e96c5769a67570c995c8cd209a89cf66` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
