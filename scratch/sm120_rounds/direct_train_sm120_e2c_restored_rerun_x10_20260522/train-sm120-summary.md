# direct_train_sm120_e2c_restored_rerun_x10_20260522

- Command: `bash train-sm120.sh`
- Active `train_gpt2cu` sha256 before run: `e2c853701f50d71e1540a520156e2bb6c7046654afb68e9bf37892cd049597ed`
- Startup: CUDA-kernel grad zero, Torch C++ dresidual zero, host scalar grad scale, fused GELU, ZeRO stage 1, BF16.
- Estimated maximum batch size: 73
- Initial val loss: 11.033154
- Final val loss: 9.483727
- Step timings ms: 2492.34, 2486.46, 2488.54, 2490.47, 2498.35, 2500.40, 2501.46, 2504.56, 2507.65, 2506.68
- Visible first-three average ms: 2489.113333
- Visible first-five average ms: 2491.232000
- Visible x10 average ms: 2497.691000
- Trainer average ms: 2498.284976
- Checkpoints written and retained: `log124M/5090_S/model_00000010.bin`, `log124M/5090_S/state_00000010_00000.bin`
- Decision: reproduction miss. Rebuilding the selected source stack changed the binary hash but not the observed performance band; this run remains slower than the user's pasted first-three and first-five timings and slower than `new-goal.md`.
