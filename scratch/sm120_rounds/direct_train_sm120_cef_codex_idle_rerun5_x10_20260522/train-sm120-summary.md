# direct_train_sm120_cef_codex_idle_rerun5_x10_20260522

- Command: `./train-sm120.sh`
- Rationale: second immediate selected-script rerun after confirming the GPU had no active processes.
- Active binary hash: `train_gpt2cu` sha256 `cef1ac4a06faee188cc46a98317f3b36b3002fffb7bdc3be10842e095797636c`.
- Startup confirmed `B=64`, `T=1024`, `total_batch_size=524288`, `grad_accum_steps=8`, `gelu_fusion=1`, `grad_zero_backend=CUDA kernel`, `dresidual_zero_backend=Torch C++`, `grad_scale_backend=host scalar`, and `zero_stage=1`.
- Device usage: `28819 MiB / 32606 MiB`; estimated maximum batch size `73`.
- Step timings: `2501.68`, `2499.65`, `2502.26`, `2506.16`, `2508.45`, `2511.15`, `2511.90`, `2513.34`, `2515.75`, `2517.51 ms`.
- Trainer average: `2509.575870 ms`.
- Visible x10 average: `2508.785000 ms`.
- Visible first-three average: `2501.196667 ms`.
- Visible first-five average: `2503.640000 ms`.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- User pasted comparison sample: first-five average `2462.236000 ms`.
- Decision: reproduction miss and slower than the prior immediate rerun. This points to run-to-run/clock-runtime variability in the same selected stack, not to a different active kernel mix.
