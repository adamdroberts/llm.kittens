# Direct `train-sm120.sh` Rerun After User Busy-GPU Note

- Date: 2026-05-22
- Command: `bash train-sm120.sh`
- Active binary: `train_gpt2cu` sha256 `e2c853701f50d71e1540a520156e2bb6c7046654afb68e9bf37892cd049597ed`
- Backend startup mix: CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar grad scale, fused GELU, ZeRO stage 1.
- Device: NVIDIA GeForce RTX 5090
- Initial val loss: `11.033154`
- Final val loss: `9.483727`

## Step Timings

| Step | ms |
| ---: | ---: |
| 1 | 2488.09 |
| 2 | 2483.44 |
| 3 | 2487.56 |
| 4 | 2489.60 |
| 5 | 2491.58 |
| 6 | 2499.00 |
| 7 | 2500.83 |
| 8 | 2503.03 |
| 9 | 2503.78 |
| 10 | 2505.35 |

## Summary

- Trainer reported average: `2496.017721 ms`
- Visible x10 average: `2495.226000 ms`
- Visible first-three average: `2486.363333 ms`
- Visible first-five average: `2488.054000 ms`
- Delta vs `new-goal.md` first-three target `2469.843 ms`: `+16.520333 ms`
- Delta vs user's pasted first-three average `2461.450 ms`: `+24.913333 ms`
- Delta vs user's pasted first-five average `2462.236 ms`: `+25.818000 ms`

## Decision

This run is faster than the immediately previous Codex direct rerun, but it
still does not reproduce the user's `~2462 ms` band or beat the historical
`new-goal.md` first-three target. Treat it as a better no-contention control,
not promotion evidence for a new kernel mix.
