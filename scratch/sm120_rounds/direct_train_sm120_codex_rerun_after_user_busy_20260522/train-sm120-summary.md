# direct_train_sm120_codex_rerun_after_user_busy_20260522

- Command: `bash train-sm120.sh`
- Active `train_gpt2cu` sha256 before run: `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`
- Startup: CUDA-kernel grad zero, Torch C++ dresidual zero, host scalar grad scale, fused GELU, ZeRO stage 1, BF16.
- Estimated maximum batch size: 73
- Initial val loss: 11.033154
- Final val loss: 9.483727
- Step timings ms: 2489.65, 2485.15, 2487.15, 2495.33, 2499.15, 2498.13, 2502.87, 2503.86, 2505.36, 2507.66
- Visible first-three average ms: 2487.316667
- Visible first-five average ms: 2491.286000
- Visible x10 average ms: 2497.431000
- Trainer average ms: 2498.294274
- Checkpoints written and retained: `log124M/5090_S/model_00000010.bin`, `log124M/5090_S/state_00000010_00000.bin`
- Decision: reproduction miss. This run is slower than the user's pasted first-three and first-five timings and slower than `new-goal.md`; it matches the recent Codex direct-script band closely enough to treat the gap as current runtime operating-point variance, not selected-stack drift.
