# SM120 Round Metrics - codex_sm120_round_libtorch_dresidual_zero_x10_20260521

- artifact dir: `scratch/sm120_rounds/codex_sm120_round_libtorch_dresidual_zero_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_libtorch_dresidual_zero_x10_20260521`
- git commit: `0f21747`

## Round Manifest

- manifest: `scratch/sm120_rounds/codex_sm120_round_libtorch_dresidual_zero_x10_20260521/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `504`

## Training Smoke

- use_master_weights: `disabled`
- gelu_fusion: `1`
- total average iteration time: `2495.957 ms`
- final val loss: `9.483727`
- final step: `10/10`, loss `9.588612`, `2505.48 ms`, `209978 tok/s`

| Step | Loss | Norm | LR | Time (ms) | Tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.032358 | 22.1414 | 8.57e-07 | 2491.31 | 210447 |
| 2 | 10.958507 | 22.0968 | 1.71e-06 | 2484.71 | 211006 |
| 3 | 10.811316 | 21.1251 | 2.57e-06 | 2486.79 | 210915 |
| 4 | 10.610130 | 18.7014 | 3.43e-06 | 2490.37 | 210779 |
| 5 | 10.392586 | 15.0184 | 4.29e-06 | 2491.97 | 210674 |
| 6 | 10.186255 | 12.0843 | 5.14e-06 | 2498.16 | 210497 |
| 7 | 10.010621 | 10.2002 | 6e-06 | 2500.26 | 210345 |
| 8 | 9.855870 | 8.7905 | 6.86e-06 | 2502.10 | 210211 |
| 9 | 9.719423 | 7.4665 | 7.71e-06 | 2503.77 | 210091 |
| 10 | 9.588612 | 6.3099 | 8.57e-06 | 2505.48 | 209978 |

