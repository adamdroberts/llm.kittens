# direct_train_sm120_cef_user_rerun3_x10_20260522

- Command: `bash train-sm120.sh`.
- Rationale: rerun the exact selected script again after the user reported their own run was likely less affected by GPU contention.
- Build/runtime settings: selected cuBLASLt-backed trainer route, current promoted attention route with default tiles, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, script-default `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2492.33, 2486.34, 2489.37, 2494.40, 2498.75, 2501.46, 2502.95, 2505.92, 2507.38, 2509.17 ms.
- Trainer average: 2499.527613 ms.
- Visible x10 average: 2498.807000 ms.
- Visible first-three average: 2489.346667 ms.
- Visible first-five average: 2492.238000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Active binary hash: `train_gpt2cu` sha256 `cef1ac4a06faee188cc46a98317f3b36b3002fffb7bdc3be10842e095797636c`.
- Checkpoints: direct script wrote and retained `log124M/5090_S/model_00000010.bin` and `log124M/5090_S/state_00000010_00000.bin`.
- Decision: reproduction miss. This run is `19.504 ms` slower than `new-goal.md` first-three, `27.897 ms` slower than the user's pasted first-three band, and `29.798 ms` slower than the user's pasted first-five band. The selected backend mix is active, so this result does not justify changing the selected kernel stack.
