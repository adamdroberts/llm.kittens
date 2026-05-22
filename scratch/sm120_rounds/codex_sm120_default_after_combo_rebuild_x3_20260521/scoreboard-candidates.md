# SM120 Round Metrics - codex_sm120_default_after_combo_rebuild_x3_20260521

- artifact dir: `scratch/sm120_rounds/codex_sm120_default_after_combo_rebuild_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_default_after_combo_rebuild_x3_20260521`
- git commit: `0f21747`

## Round Manifest

- manifest: `scratch/sm120_rounds/codex_sm120_default_after_combo_rebuild_x3_20260521/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `514`

## Training Smoke

- use_master_weights: `disabled`
- gelu_fusion: `1`
- total average iteration time: `2496.877 ms`
- final val loss: `10.609911`
- final step: `3/3`, loss `10.811316`, `2498.64 ms`, `209974 tok/s`

| Step | Loss | Norm | LR | Time (ms) | Tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.032358 | 22.1414 | 8.57e-07 | 2500.88 | 209641 |
| 2 | 10.958507 | 22.0968 | 1.71e-06 | 2495.12 | 210126 |
| 3 | 10.811316 | 21.1251 | 2.57e-06 | 2498.64 | 209974 |

