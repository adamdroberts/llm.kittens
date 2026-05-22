# Direct train-sm120 after fcproj restore x10 2026-05-22

- Command: `./train-sm120.sh`
- Binary: `train_gpt2cu` sha256 `2032b2086406ee051fa8f18a5ae2c716c70e6d8165f4bc5383fb24e7c7657ac1`
- Startup: `B=64`, `T=1024`, `total_batch_size=524288`, `grad_accum_steps=8`, `gelu_fusion=1`, `grad_zero_backend=CUDA kernel`, `dresidual_zero_backend=Torch C++`, `grad_scale_backend=host scalar`, `zero_stage=1`
- Device usage: `28819 MiB / 32606 MiB`, estimated maximum batch size `73`
- Step timings: `2517.96`, `2517.30`, `2518.46`, `2524.09`, `2589.31`, `2693.36`, `2650.45`, `2549.43`, `2554.61`, `2555.64 ms`
- Trainer average: `2572.516229 ms`
- Visible x10 average: `2567.061000 ms`
- Visible first-three average: `2517.906667 ms`
- Visible first-five average: `2533.424000 ms`
- Losses: `11.033154 -> 9.483727`
- Checkpoints: direct script wrote and retained `log124M/5090_S/model_00000010.bin` and `log124M/5090_S/state_00000010_00000.bin`
- Post-run GPU snapshot: `P8`, `39 C`, `1%` GPU utilization, `2063 MiB / 32607 MiB`, graphics clock `367 MHz`, memory clock `405 MHz`
- Decision: reproduction miss. The restored selected stack was active, but this run is slower than the user's pasted first-five average `2462.236000 ms` and slower than the current selected x10 control.
