# SM120 Round Metrics - codex_sm120_libtorch_grad_zero_current_refresh_x10_20260522

- artifact dir: `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_current_refresh_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_libtorch_grad_zero_current_refresh_x10_20260522`
- git commit: `0f21747`

## Round Manifest

- manifest: `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_current_refresh_x10_20260522/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `683`

## Training Smoke

- use_master_weights: `disabled`
- gelu_fusion: `1`
- total average iteration time: `2505.490 ms`
- final val loss: `9.483727`
- final step: `10/10`, loss `9.588612`, `2515.33 ms`, `209179 tok/s`

| Step | Loss | Norm | LR | Time (ms) | Tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.032358 | 22.1414 | 8.57e-07 | 2497.53 | 209923 |
| 2 | 10.958507 | 22.0968 | 1.71e-06 | 2490.42 | 210522 |
| 3 | 10.811316 | 21.1251 | 2.57e-06 | 2497.78 | 210204 |
| 4 | 10.610130 | 18.7014 | 3.43e-06 | 2502.52 | 209959 |
| 5 | 10.392586 | 15.0184 | 4.29e-06 | 2504.86 | 209783 |
| 6 | 10.186255 | 12.0843 | 5.14e-06 | 2506.18 | 209654 |
| 7 | 10.010621 | 10.2002 | 6e-06 | 2507.94 | 209540 |
| 8 | 9.855870 | 8.7905 | 6.86e-06 | 2511.05 | 209416 |
| 9 | 9.719423 | 7.4665 | 7.71e-06 | 2513.33 | 209295 |
| 10 | 9.588612 | 6.3099 | 8.57e-06 | 2515.33 | 209179 |

