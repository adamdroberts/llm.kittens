# Direct `train-sm120.sh` Post-Restore Control

- Date: 2026-05-22
- Command: `bash train-sm120.sh`
- Active binary: `train_gpt2cu` sha256 `4374a593bdc123692c38f28c92951abbe914c4f56f96a9fd4e140fa3c4b119da`
- Backend startup mix: CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar grad scale, fused GELU, ZeRO stage 1.
- Device: NVIDIA GeForce RTX 5090
- Initial val loss: `11.033154`
- Final val loss: `9.483727`

## Step Timings

| Step | ms |
| ---: | ---: |
| 1 | 2490.11 |
| 2 | 2484.31 |
| 3 | 2487.35 |
| 4 | 2491.20 |
| 5 | 2495.80 |
| 6 | 2501.34 |
| 7 | 2504.18 |
| 8 | 2507.46 |
| 9 | 2507.20 |
| 10 | 2508.05 |

## Summary

- Trainer reported average: `2498.544534 ms`
- Visible x10 average: `2497.700000 ms`
- Visible first-three average: `2487.256667 ms`
- Visible first-five average: `2489.754000 ms`
- Delta vs `new-goal.md` first-three target `2469.843 ms`: `+17.413667 ms`
- Delta vs user's pasted first-three average `2461.450 ms`: `+25.806667 ms`
- Delta vs user's pasted first-five average `2462.236 ms`: `+27.518000 ms`

## Decision

This restored selected-stack control stays in the recent slower direct-script
band. It confirms the rejected TK approximate dGELU candidate was removed, but
it does not reproduce the user's faster `~2462 ms` band or provide new
promotion evidence.
