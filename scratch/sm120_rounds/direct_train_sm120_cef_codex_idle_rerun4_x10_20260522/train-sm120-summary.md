# direct_train_sm120_cef_codex_idle_rerun4_x10_20260522

- Command: `./train-sm120.sh`
- Rationale: rerun the exact selected script after the user pasted a faster no-contention sample and suggested the prior Codex run may have hit a busy GPU.
- Active binary hash: `train_gpt2cu` sha256 `cef1ac4a06faee188cc46a98317f3b36b3002fffb7bdc3be10842e095797636c`.
- Startup confirmed `B=64`, `T=1024`, `total_batch_size=524288`, `grad_accum_steps=8`, `gelu_fusion=1`, `grad_zero_backend=CUDA kernel`, `dresidual_zero_backend=Torch C++`, `grad_scale_backend=host scalar`, and `zero_stage=1`.
- Device usage: `28819 MiB / 32606 MiB`; estimated maximum batch size `73`.
- Step timings: `2493.99`, `2488.06`, `2489.74`, `2495.45`, `2499.81`, `2501.69`, `2505.13`, `2506.93`, `2509.96`, `2509.84 ms`.
- Trainer average: `2500.735177 ms`.
- Visible x10 average: `2500.060000 ms`.
- Visible first-three average: `2490.596667 ms`.
- Visible first-five average: `2493.410000 ms`.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Post-run GPU snapshot: `nvidia-smi` reported no running GPU processes, P8 idle state, `798 MiB / 32607 MiB`, and `1%` GPU utilization.
- User pasted comparison sample: first five steps `2464.71`, `2456.65`, `2462.99`, `2461.49`, `2465.34 ms`; first-five average `2462.236000 ms`.
- Decision: reproduction miss. This run is `31.174 ms` slower than the user's pasted first-five average and does not justify a kernel-stack change.
