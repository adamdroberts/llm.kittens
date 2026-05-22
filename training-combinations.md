# SM120 Training Combination Trials

This file tracks end-to-end GPT-2 TinyStories training trials for SM120 / RTX
5090. A kernel stack or setting is not considered promoted just because a
focused benchmark wins; it needs trainer evidence against the current stable
baseline.

## Targets

| Target | Evidence | Average step time | Notes |
| --- | --- | ---: | --- |
| Historical best note | `new-goal.md` first three reported steps | 2469.843 ms | Target to reproduce or beat. |
| Current stable baseline | `codex_sm120_round_backward_stream_sync_default_x10_20260521` | 2493.133 ms | Current stable x10 baseline to beat before promotion. |
| Promoted SM120 fast default | direct `./train-sm120.sh` after source promotion | 2489.062 ms | Current direct-script best with CUDA grad-zero, Torch C++ dresidual-zero, dprep=3, block1024, LayerNorm bwd1, and `CUDA_DEVICE_MAX_CONNECTIONS=1`. |

## Trial Queue

| Trial | Status | Kernel settings | Training evidence | Decision |
| --- | --- | --- | --- | --- |
| `current_stable_default_x10_20260521` | reference | Current native trainer mix, no LibTorch trainer memory route, default bias-add blocks | x10 avg 2493.133 ms | Baseline. |
| `default_after_combo_rebuild_x3_20260521` | reference | Rebuilt default binary after rejected candidate trials | `scratch/sm120_rounds/codex_sm120_default_after_combo_rebuild_x3_20260521`, x3 avg 2496.877 ms | Confirms `train_gpt2cu` is back on the default runtime path. |
| `direct_train_sm120_default_x10_20260521` | reference | Direct `./train-sm120.sh` after clean default rebuild | `log124M/5090_S/main.log`, x10 avg 2507.656 ms | Confirms the user-facing script path is currently slower than both the stable harness baseline and the historical `new-goal.md` note. |
| `direct_train_sm120_current_rerun_x10_20260521` | regression reference | Direct `./train-sm120.sh` with current default binary `e7abfa54...` before fresh rebuild | `scratch/train-sm120-current-rerun-20260521.log`, x10 avg 2656.684 ms | Current user-facing script path is now much slower even though startup still reports CUDA runtime zeroing and `gelu_fusion=1`. |
| `direct_train_sm120_after_wide_reset_x10_20260521` | reference | Direct `./train-sm120.sh` after rebuilding default binaries without rejected candidate flags | `scratch/train-sm120-default-after-wide-reset-x10-20260521.log`, x10 avg 2510.584 ms | Severe slowdown recovered, but the direct script path remains slower than the stable harness baseline. |
| `direct_train_sm120_after_profiler_rebuild_x10_20260521` | regression reference | Direct `./train-sm120.sh` after rebuilding default binaries with profiler guard source present but no candidate flags | `scratch/train-sm120-direct-after-profiler-rebuild-x10-20260521.log`, x10 avg 2519.472 ms | Confirms the direct script remains slower with the default CUDA runtime zeroing path and `gelu_fusion=1`. |
| `direct_train_sm120_fresh_default_after_restore_x10_20260521` | reference | Direct `./train-sm120.sh` after restoring default/no-candidate binaries following the rejected profiler x10 run | `log124M/5090_S`, x10 avg 2508.204 ms | Confirms the current direct script recovered from the severe 2656.684 ms regression band but still trails the stable harness baseline by 15.071 ms. |
| `direct_train_sm120_with_dmon_x10_20260521` | regression reference | Direct `./train-sm120.sh` with concurrent `nvidia-smi dmon` telemetry | `scratch/train-sm120-direct-with-dmon-x10-20260521.log`, x10 avg 2529.901 ms; telemetry `scratch/dmon-train-sm120-direct-20260521.log` | Kernel stack unchanged; dmon shows full SM utilization, ~574-580 W, 62-69 C, no power/thermal violation flags, and pclk mostly 2647-2685 MHz during the slow run. |
| `direct_train_sm120_live_control_x10_20260521` | regression reference | Direct `./train-sm120.sh` on restored default/no-candidate binary hash `88b5510f...` | live direct run, x10 avg 2629.656 ms | Reproduces the current slow band while startup still reports CUDA runtime zeroing, host scalar grad scale, and fused GELU. This is not a Torch/LibTorch route or promoted candidate stack. |
| `libtorch_grad_zero_live_retest_x3_20260521` | rejected-current-state-win | `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0` | `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_live_retest_x3_20260521`, x3 avg 2600.170 ms | Correctness passed and it beat the same-session restored default x3 by 15.655 ms, but it is still 107.037 ms slower than the stable x10 baseline and the prior LibTorch grad-zero x10 route already regressed. Do not promote. |
| `default_after_libtorch_grad_zero_live_retest_x3_20260521` | reference | Restored default/no-candidate binary after the LibTorch grad-zero live retest | `log124M/5090_S_default_after_libtorch_grad_zero_live_retest_x3_20260521`, x3 avg 2615.825 ms | Confirms the binary was restored to CUDA runtime zeroing and gives a same-session current-state reference for the Torch retest. |
| `combo_libtorch_grad_zero_precompute_scale_x3_20260521` | rejected | `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_precompute_scale_x3_20260521`, x3 avg 2516.894 ms | Correctness passed, but the restored same-session default x3 averaged 2502.819 ms, so this composition is slower by 14.075 ms and still trails the stable x10 baseline by 23.761 ms. |
| `default_after_combo_libtorch_grad_zero_precompute_scale_x3_20260521` | reference | Restored default/no-candidate binary after the LibTorch grad-zero plus precomputed-grad-scale combo | `log124M/5090_S_default_after_combo_libtorch_grad_zero_precompute_scale_x3_20260521`, x3 avg 2502.819 ms | Same-session control shows the combo result was not a real improvement over restored default. |
| `combo_libtorch_grad_zero_layernorm_bwd1_x3_20260521` | rejected-current-state-win | `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_layernorm_bwd1_x3_20260521`, x3 avg 2549.664 ms | Correctness passed and it beat the degraded same-session default x3 by 48.017 ms, but it still trails the stable x10 baseline by 56.531 ms and the LibTorch grad-zero route already failed x10 stability. Do not promote. |
| `default_after_combo_libtorch_grad_zero_layernorm_bwd1_x3_20260521` | reference | Restored default/no-candidate binary after the LibTorch grad-zero plus LayerNorm bwd1 combo | `log124M/5090_S_default_after_combo_libtorch_grad_zero_layernorm_bwd1_x3_20260521`, x3 avg 2597.681 ms | Same-session control is degraded, so the combo is only a current-state relative win, not a stable speed improvement. |
| `combo_libtorch_dresidual_zero_bias_wide1024_x3_20260521` | rejected-noise-win | `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `SM120_USE_LIBTORCH_GRAD_ZERO=0`, `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_bias_wide1024_x3_20260521`, x3 avg 2509.732 ms | Correctness passed and it beat the same-session default x3 by 1.349 ms, but it is still 16.599 ms slower than the stable x10 baseline and both component routes already failed stability gates. Do not promote. |
| `default_after_combo_libtorch_dresidual_zero_bias_wide1024_x3_20260521` | reference | Restored default/no-candidate binary after the LibTorch dresidual-zero plus wide-bias combo | `log124M/5090_S_default_after_combo_libtorch_dresidual_zero_bias_wide1024_x3_20260521`, x3 avg 2511.081 ms | Same-session control shows only a noise-level candidate edge and confirms default restoration. |
| `combo_bias_wide1024_layernorm_bwd1_x3_20260521` | rejected-same-session-win | `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` | `scratch/sm120_rounds/codex_sm120_combo_bias_wide1024_layernorm_bwd1_x3_20260521`, x3 avg 2499.946 ms | Correctness passed and it beat the same-session default x3 by 3.385 ms, but it still trails the stable x10 baseline by 6.813 ms and both component routes already failed x10 stability. Do not promote. |
| `default_after_combo_bias_wide1024_layernorm_bwd1_x3_20260521` | reference | Restored default/no-candidate binary after the native wide-bias plus LayerNorm bwd1 combo | `log124M/5090_S_default_after_combo_bias_wide1024_layernorm_bwd1_x3_20260521`, x3 avg 2503.331 ms | Same-session control confirms a small candidate edge, but the candidate remains behind the stable baseline. |
| `combo_project_fastest_dinp_tk_dgelu_x3_20260521` | rejected-same-session-win | `LLMK_SM120_USE_CUBLAS_DINP_ATTPROJ`, `LLMK_SM120_USE_CUBLAS_DINP_FC`, `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` | `scratch/sm120_rounds/codex_sm120_combo_project_fastest_dinp_tk_dgelu_x3_20260521`, x3 avg 2498.856 ms | Correctness and validation passed, and this direct composition of fastest trainer-callable dInput/dGeLU microbench rows beat the same-session restored default x3 by 7.861 ms. It still trails the stable x10 baseline by 5.723 ms, so do not promote. |
| `default_after_combo_project_fastest_dinp_tk_dgelu_rerun_x3_20260521` | reference | Restored default/no-candidate binary after the combined fastest-row dInput/dGeLU candidate | `scratch/train-sm120-default-after-combo-project-fastest-dinp-tk-dgelu-rerun-x3-20260521.log`, x3 avg 2506.717 ms | Same-session control confirms the candidate has a current-state edge, but the restored default remains slower than the stable baseline under the current runtime state. |
| `combo_all_native_winners_x3_20260521` | rejected-same-session-win | `LLMK_SM120_USE_CUBLAS_DINP_ATTPROJ`, `LLMK_SM120_USE_CUBLAS_DINP_FC`, `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024` | `scratch/sm120_rounds/codex_sm120_combo_all_native_winners_x3_20260521`, x3 avg 2498.178 ms | Correctness and validation passed. This broader native composition improved 0.678 ms over the narrower fastest dInput/dGeLU combo and beat the same-session restored default x3 by 8.539 ms, but it still trails the stable x10 baseline by 5.045 ms. Do not promote. |
| `combo_libtorch_grad_zero_all_native_winners_x3_20260521` | rejected-near-miss | `SM120_USE_LIBTORCH_GRAD_ZERO=1` plus all-native winner flags: direct cuBLAS dInput for attproj/MLP-up, TK fused dGELU dInput, LayerNorm bwd1, and wide bias-add | `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_all_native_winners_x3_20260521`, x3 avg 2497.004 ms | Correctness passed and this improves 1.174 ms over the all-native composition, but it still trails the stable x10 baseline by 3.871 ms. Do not promote; default binary restored. |
| `combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_x3_20260521` | rejected-near-miss | `SM120_USE_LIBTORCH_GRAD_ZERO=1` plus direct cuBLAS dInput for attproj/MLP-up and TK fused dGELU dInput | `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_x3_20260521`, x3 avg 2496.595 ms | Correctness passed and this is the best composed x3 result so far, but it remains 3.462 ms slower than the stable x10 baseline. Do not promote; default binary restored. |
| `combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_maxconn1_x3_20260521` | rejected | `SM120_USE_LIBTORCH_GRAD_ZERO=1` plus fastest dInput/dGELU stack and `CUDA_DEVICE_MAX_CONNECTIONS=1` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_maxconn1_x3_20260521`, x3 avg 2497.422 ms | Correctness passed, but adding maxconn1 worsened the best dInput/dGELU stack by 0.828 ms and still trailed stable x10 by 4.289 ms. Do not promote. |
| `combo_libtorch_grad_zero_tk_dgelu_x3_20260521` | rejected | `SM120_USE_LIBTORCH_GRAD_ZERO=1` plus `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`, without direct-cuBLAS dInput selectors | `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_tk_dgelu_x3_20260521`, x3 avg 2658.973 ms | Correctness passed and Torch C++ grad-zero was active, but isolating TK dGELU with Torch grad-zero regressed by 165.840 ms versus stable x10. Do not promote. |
| `libtorch_grad_zero_maxconn1_x3_20260521` | rejected-after-x10 | `SM120_USE_LIBTORCH_GRAD_ZERO=1`, native dInput/dGELU defaults, `CUDA_DEVICE_MAX_CONNECTIONS=1` | `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_maxconn1_x3_20260521`, x3 avg 2492.843 ms | Short run beat the stable x10 baseline by 0.290 ms, so it was gated with x10 before promotion. |
| `libtorch_grad_zero_maxconn1_x10_20260521` | rejected-x10 | Same stack as `libtorch_grad_zero_maxconn1_x3_20260521` | `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_maxconn1_x10_20260521`, x10 avg 2497.047 ms | Longer gate rejected the short-run candidate: it is 3.914 ms slower than stable x10. Default binary restored to `cbcf72b7...`. |
| `current_default_refresh_x10_20260521` | regression reference | Fresh default harness build, no candidate flags, no LibTorch trainer route | `scratch/sm120_rounds/codex_sm120_current_default_refresh_x10_20260521`, x10 avg 2627.950 ms | Current machine/runtime state is much slower even on the default backend mix, so do not treat later candidate regressions against this band as kernel-specific proof. |
| `restored_default_hash_x3_20260521` | regression reference | Exact default/no-candidate restore build with known `train_gpt2cu` hash `dba87...` | `log124M/5090_S_codex_sm120_restored_default_hash_x3_20260521`, x3 avg 2694.590 ms | Confirms the slowdown persists after restoring the known default artifact; this is not a leftover candidate flag. |
| `default_dmon_probe_x3_20260521` | regression reference | Same restored default hash with concurrent `nvidia-smi dmon` telemetry | `log124M/5090_S_codex_sm120_default_dmon_probe_x3_20260521`, x3 avg 2624.396 ms | dmon showed 99-100% SM utilization, ~574-577 W, 58-63 C, pclk ~2707-2775 MHz during the slow run, and no power/thermal violation flags. |
| `current_control_fresh_rebuild_x3_20260521` | noisy reference | Fresh default rebuild, unique harness output dir, no candidate flags | `scratch/sm120_rounds/codex_sm120_current_control_x3_20260521`, x3 avg 2583.795 ms | Later default recheck recovered, so this is not treated as stable current speed evidence. The earlier `fcproj dInp+dGeLU` 5009.99 us spot-check was not reproduced under a clean default `bench_sm120_matmul`. |
| `default_post_bench_reset_x3_20260521` | reference | Rebuilt default trainer and default benchmark artifacts after rejected candidates | `scratch/train-sm120-default-post-bench-reset-x3-20260521.log`, x3 avg 2498.128 ms | Confirms current default training speed is back near the stable baseline, and the clean default matmul recheck has `fcproj dInp+dGeLU` cuBLASLt fused at 1859.02 us. |
| `default_native_benchmark_warmup_x3_20260521` | reference/rejected-warmup | Current default/no-candidate binary, correctness and native benchmark phases run before training | `scratch/sm120_rounds/codex_sm120_default_native_benchmark_warmup_x3_20260521`, x3 avg 2499.009 ms | Does not reproduce the older `memory_shape_coverage` 2490.206 ms x3 run and remains 5.876 ms slower than the stable x10 baseline. Keep as drift evidence, not a promotion. |
| `live_default_control_x3_20260521` | regression reference | Current restored default binary, no candidate flags, no CUDA scheduling override | `scratch/train-sm120-live-default-control-x3-20260521.log`, x3 avg 2527.266 ms | Current live runtime band is slower than the stable x10 baseline while startup still reports CUDA runtime zeroing, host scalar grad scale, and fused GELU. |
| `cuda_maxconn1_x3_20260521` | recovery reference | Current restored default binary with `CUDA_DEVICE_MAX_CONNECTIONS=1` | `scratch/train-sm120-cuda-maxconn1-x3-20260521.log`, x3 avg 2495.607 ms | Recovers 31.659 ms versus the same-session live default x3, but still trails the stable x10 baseline and needs longer gating. |
| `cuda_maxconn1_x10_20260521` | rejected-runtime-knob | Current restored default binary with `CUDA_DEVICE_MAX_CONNECTIONS=1` | `scratch/train-sm120-cuda-maxconn1-x10-20260521.log`, x10 avg 2510.431 ms | Longer gate rejects this as a promotion setting: it improves the degraded live band but is 17.298 ms slower than the stable x10 baseline. |
| `cuda_maxconn32_x3_20260521` | rejected | Current restored default binary with `CUDA_DEVICE_MAX_CONNECTIONS=32` | `scratch/train-sm120-cuda-maxconn32-x3-20260521.log`, x3 avg 2512.792 ms | Worse than `CUDA_DEVICE_MAX_CONNECTIONS=1` and slower than the stable baseline. |
| `combo_all_native_winners_maxconn1_x3_20260521` | rejected | Best all-native trainer-callable composition plus `CUDA_DEVICE_MAX_CONNECTIONS=1` | `scratch/train-sm120-combo-all-native-winners-maxconn1-x3-20260521.log`, x3 avg 2506.429 ms | Correctness passed, but combining the native winner stack with maxconn1 regressed versus maxconn1 default and remains slower than the stable x10 baseline. |
| `combo_libtorch_dresidual_zero_maxconn1_x3_20260521` | rejected | `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `SM120_USE_LIBTORCH_GRAD_ZERO=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_maxconn1_x3_20260521`, x3 avg 2500.425 ms | Correctness passed and the Torch C++ dresidual-zero route was active, but the combo is slower than both the stable x10 baseline and the maxconn1 default x3. Do not promote. |
| `combo_libtorch_dresidual_zero_fastest_dinp_tk_dgelu_x3_20260521` | rejected | `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1` plus direct cuBLAS dInput for attproj/MLP-up and TK fused dGELU dInput | `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_fastest_dinp_tk_dgelu_x3_20260521`, x3 avg 2510.396 ms | Correctness passed, but combining the Torch residual clear with the fastest dInput/dGeLU stack regressed versus both component evidence and stable x10. |
| `combo_libtorch_grad_zero_bias_wide1024_x3_20260521` | rejected | `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0`, `EXTRA_NVCC_FLAGS=-DLLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_bias_wide1024_x3_20260521`, x3 avg 2688.174 ms | Correctness and validation passed, but training regressed by 195.041 ms versus stable x10 baseline. Do not promote the LibTorch grad-zero trainer route. |
| `combo_libtorch_grad_zero_dresidual_zero_bias_wide1024_x3_20260521` | skipped | Add `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1` to the first combo | not run | Skipped because the narrower LibTorch grad-zero combination was already a large end-to-end regression. |
| `combo_cublas_dinp_fc_bias_wide1024_x3_20260521` | rejected | `LLMK_SM120_USE_CUBLAS_DINP_FC`, `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024` | `scratch/sm120_rounds/codex_sm120_combo_cublas_dinp_fc_bias_wide1024_x3_20260521`, x3 avg 2600.928 ms | Correctness and validation passed, but training regressed by 107.795 ms versus stable x10 baseline. Do not promote this composition. |
| `combo_layernorm_bwd1_classifier_exp2_x3_20260521` | rejected | `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_SM120_CLASSIFIER_EXP2` | `scratch/sm120_rounds/codex_sm120_combo_layernorm_bwd1_classifier_exp2_x3_20260521`, x3 avg 2504.913 ms | Correctness and validation passed, but training regressed by 11.780 ms versus stable x10 baseline. Do not promote this composition. |
| `combo_cublas_dinp_attproj_layernorm_bwd1_x3_20260521` | rejected | `LLMK_SM120_USE_CUBLAS_DINP_ATTPROJ`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` | `scratch/sm120_rounds/codex_sm120_combo_cublas_dinp_attproj_layernorm_bwd1_x3_20260521`, x3 avg 2507.472 ms | Correctness and validation passed, but training regressed by 14.339 ms versus stable x10 baseline. Do not promote this composition. |
| `combo_tk_dgelu_layernorm_bwd1_x3_20260521` | rejected | `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` | `scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521`, x3 avg 2527.814 ms | Correctness and validation passed, but training regressed by 34.681 ms versus stable x10 baseline. Do not promote this composition. |
| `tk_dgelu_current_retest_x3_20260521` | rejected | `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` only, retested after current cuBLASLt fused focused-row regression | `scratch/sm120_rounds/codex_sm120_tk_dgelu_current_retest_x3_20260521`, x3 avg 2672.022 ms | Rejected again. Even with current focused cuBLASLt fused timing looking bad, the isolated TK fused dGELU trainer route is slower than the fresh default control. |
| `ge0_current_x3_20260521` | rejected | Runtime `-ge 0`, explicit GELU forward/backward route | `scratch/train-sm120-ge0-current-x3-20260521.log`, x3 avg 2622.199 ms | Rejected. Disabling the fused GELU route is much slower than the current default. |
| `bias_vec4_retest_x3_20260521` | rejected | Opt-in `LLMK_SM120_BIAS_ADD_VEC4` compile flag | `scratch/train-sm120-bias-vec4-retest-x3-20260521.log`, x3 avg 2508.513 ms | Rejected. Correctness passed, but focused bias-add rows and trainer timing were worse than default. |
| `combo_libtorch_grad_dresidual_zero_x3_20260521` | rejected | `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_dresidual_zero_x3_20260521`, x3 avg 2674.668 ms | Correctness and validation passed, but training regressed by 181.535 ms versus stable x10 baseline. Do not promote LibTorch memory routes together. |
| `bias_wide1024_x10_20260521` | rejected | `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024` | `scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521`, x10 avg 2509.814 ms | Slower than the 2493.133 ms stable x10 baseline. The earlier x3 wide-block signal does not survive a stability run. |
| `cuda_kernel_zero_both_x3_20260521` | rejected | `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, `LLMK_SM120_USE_CUDA_KERNEL_DRESIDUAL_ZERO` | `scratch/sm120_rounds/codex_sm120_cuda_kernel_zero_both_x3_20260521_escalated`, x3 avg 2499.725 ms | Trainer-callable native CUDA-kernel zero route worked, but did not beat the stable baseline or fresh default reference. |
| `default_after_cuda_kernel_zero_hook_x3_20260521` | reference | Unflagged default after adding opt-in CUDA-kernel-zero hooks | `scratch/sm120_rounds/codex_sm120_default_after_cuda_kernel_zero_hook_x3_20260521`, x3 avg 2499.649 ms | Confirms the new hooks do not change the default zeroing path. |
| `cuda_kernel_zero_both_block1024_x3_20260521` | rejected | `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, `LLMK_SM120_USE_CUDA_KERNEL_DRESIDUAL_ZERO`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024` | `scratch/sm120_rounds/codex_sm120_cuda_kernel_zero_both_block1024_x3_20260521`, x3 avg 2498.795 ms | Small same-session x3 edge over the unflagged hook build, but still slower than the 2493.133 ms stable x10 baseline and not a significant promotion signal. |
| `cublaslt_plan_cache_x3_20260521` | rejected | `LLMK_SM120_CACHE_CUBLASLT_PLANS` | `scratch/sm120_rounds/codex_sm120_cublaslt_plan_cache_x3_20260521`, x3 avg 2505.027 ms | Trainer-wide cuBLASLt plan caching regressed versus the default path, so do not promote. |
| `cublaslt_ws256_heur16_x3_20260521` | rejected | `LLMK_SM120_CUBLASLT_WORKSPACE_MB=256`, `LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS=16` | `scratch/sm120_rounds/codex_sm120_cublaslt_ws256_heur16_x3_20260521`, x3 avg 2509.786 ms | Correctness passed, but widening cuBLASLt workspace/search breadth regressed by 16.653 ms versus stable x10. Keep default 128 MB / 8-result lowest-waves selection. |
| `disable_cuda_profiler_x3_20260521` | rejected | `LLMK_DISABLE_CUDA_PROFILER` | `scratch/sm120_rounds/codex_sm120_disable_cuda_profiler_x3_20260521`, x3 avg 2498.197 ms | Profiler-disable guard has a tiny same-session edge, but is still slower than the 2493.133 ms stable x10 baseline and not enough for promotion. |
| `cublas_dinp_qkv_x3_20260521` | rejected | `LLMK_SM120_USE_CUBLAS_DINP_QKV` | `scratch/sm120_rounds/codex_sm120_cublas_dinp_qkv_x3_20260521`, x3 avg 2513.647 ms | The qkv dInput cuBLAS selector does not improve training; it is slower than the same-session default and the stable x10 baseline. |
| `default_after_qkv_selector_x3_20260521` | reference | Unflagged default after the qkv selector test | `scratch/sm120_rounds/codex_sm120_default_after_qkv_selector_x3_20260521`, x3 avg 2512.286 ms | Restores default no-flag binaries and shows current same-session drift is slow, but still slightly faster than the qkv selector. |
| `combo_cublas_fc_tk_dgelu_x3_20260521` | rejected | `LLMK_SM120_USE_CUBLAS_DINP_FC`, `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` | `scratch/sm120_rounds/codex_sm120_combo_cublas_fc_tk_dgelu_x3_20260521`, x3 avg 2574.852 ms | Combining the MLP-up cuBLAS dInput selector with TK fused dGELU regressed badly, so do not compose these MLP backward near-misses. |
| `disable_cuda_profiler_x10_20260521` | rejected | `LLMK_DISABLE_CUDA_PROFILER` | `scratch/sm120_rounds/codex_sm120_disable_cuda_profiler_x10_20260521`, x10 avg 2559.420 ms | Longer gate rejects the profiler-disable hypothesis; late-step drift remains and training is much slower than the stable x10 baseline. |
| `attention_atomic_dq_x3_20260521` | rejected | `LLMK_SM120_ATOMIC_DQ` | `scratch/sm120_rounds/codex_sm120_attention_atomic_dq_x3_20260521`, x3 avg 2945.101 ms | Correctness passed, but the trainer-shaped atomic-dQ attention route disables the packed-QKV fast path and regresses badly. |
| `combo_libtorch_dresidual_zero_layernorm_bwd1_x3_20260521` | rejected | `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_layernorm_bwd1_x3_20260521`, x3 avg 2498.079 ms | This was a real composition of two prior near-misses, but it still trailed the stable x10 baseline by 4.946 ms. Do not promote. |
| `combo_cublas_dinp_fc_layernorm_bwd1_x3_20260521` | rejected | `LLMK_SM120_USE_CUBLAS_DINP_FC`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` | `scratch/sm120_rounds/codex_sm120_combo_cublas_dinp_fc_layernorm_bwd1_x3_20260521`, x3 avg 2515.866 ms | Correctness passed, but combining the strongest prior FC dInput smoke with the LayerNorm backward candidate regressed by 22.733 ms versus stable x10. |
| `nonblocking_main_stream_x3_20260521` | rejected-correctness | `LLMK_SM120_NONBLOCKING_MAIN_STREAM` | `scratch/sm120_rounds/codex_sm120_nonblocking_main_stream_x3_20260521`, x3 avg 2619.052 ms | Focused smokes passed, but trainer semantics broke: initial val loss was 0.002881, final val loss 0.000000, and grad norms collapsed to ~0.36. Do not promote. |
| `async_grad_norm_copy_x3_20260521` | rejected | `LLMK_SM120_ASYNC_GRAD_NORM_COPY` | `scratch/sm120_rounds/codex_sm120_async_grad_norm_copy_x3_20260521`, x3 avg 2629.395 ms | Correctness and trainer losses/norms passed, but replacing the blocking grad-norm scalar copy with stream-scoped async copy plus stream sync regressed badly. |
| `device_grad_scale_adamw_x3_20260521` | rejected | `LLMK_SM120_DEVICE_GRAD_SCALE_ADAMW` | `scratch/sm120_rounds/codex_sm120_device_grad_scale_adamw_x3_20260521`, x3 avg 2627.288 ms | Correctness and trainer losses/norms passed, and the run used the device AdamW scalar path, but it still regressed by 134.155 ms versus stable x10. |
| `precompute_grad_scale_adamw_x3_20260521` | rejected | `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` | `scratch/sm120_rounds/codex_sm120_precompute_grad_scale_adamw_x3_20260521`, x3 avg 2614.956 ms | Correctness and trainer losses/norms passed, and this improved over the per-thread device-sqrt variant, but it still regressed by 121.823 ms versus stable x10. |
| `guard_adamw_helpers_default_x3_20260521` | rejected-source-size-hypothesis | Default backend mix after temporarily guarding unused opt-in AdamW helper kernels out of default builds | `scratch/train-sm120-guard-adamw-helpers-default-x3-20260521.log`, x3 avg 2617.405 ms | Startup stayed on CUDA runtime zeroing and host scalar grad scale, but training regressed by 124.272 ms versus stable x10. Patch was reverted and default binary restored. |
| `combo_cuda_zero_block1024_layernorm_bwd1_x3_20260521` | rejected | `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, `LLMK_SM120_USE_CUDA_KERNEL_DRESIDUAL_ZERO`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` | `scratch/sm120_rounds/codex_sm120_combo_cuda_zero_block1024_layernorm_bwd1_x3_20260521`, x3 avg 2617.466 ms | Correctness and trainer losses/norms passed, but combining the native CUDA zeroing near-miss with LayerNorm bwd1 did not beat the stable baseline or the precomputed-grad-scale rejected variant. |
| `cublas_dinp_fcproj_x3_20260521` | rejected | `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` | `scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_x3_20260521`, x3 avg 2651.203 ms | Correctness and trainer losses/norms passed, but forcing direct cuBLAS for the context-sensitive MLP-projection dInput row was a large trainer regression. |
| `attention_dprep3_current_x3_20260521` | rejected-near-miss | `LLMK_SM120_DPREP_WARPS=3` on the current cuBLASLt-backed trainer stack | `scratch/sm120_rounds/codex_sm120_attention_dprep3_current_x3_20260521`, x3 avg 2494.257 ms | Correctness passed and losses were normal, but the old dprep=3 signal is 1.124 ms slower than the stable x10 baseline in the current stack. No x10 gate; default binary restored. |
| `combo_cublas_dinp_attproj_fc_dprep3_x3_20260521` | rejected | `LLMK_SM120_USE_CUBLAS_DINP_ATTPROJ`, `LLMK_SM120_USE_CUBLAS_DINP_FC`, `LLMK_SM120_DPREP_WARPS=3` | `scratch/sm120_rounds/codex_sm120_combo_cublas_dinp_attproj_fc_dprep3_x3_20260521`, x3 avg 2543.476 ms | Correctness passed, but combining the closest x10 cuBLAS dInput selector near-miss with the dprep=3 attention near-miss regressed by 50.343 ms versus stable x10. Do not promote. |
| `combo_libtorch_grad_zero_maxconn1_dprep3_x3_20260521` | rejected-after-x10 | `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_maxconn1_dprep3_x3_20260521`, x3 avg 2486.639 ms | Short run beat the stable x10 baseline by 6.494 ms, so it was gated with x10 before promotion. |
| `combo_libtorch_grad_zero_maxconn1_dprep3_x10_20260521` | rejected-x10 | Same stack as `combo_libtorch_grad_zero_maxconn1_dprep3_x3_20260521` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_maxconn1_dprep3_x10_20260521`, x10 avg 2509.052 ms | Longer gate rejected the short-run candidate: late steps drifted to 2545.70 and 2562.27 ms, making it 15.919 ms slower than stable x10. Default binary restored to `cbcf72b7...`. |
| `combo_libtorch_grad_zero_dprep3_x3_20260521` | rejected-after-x10 | `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `LLMK_SM120_DPREP_WARPS=3` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_dprep3_x3_20260521`, x3 avg 2485.322 ms | Short run beat stable x10 by 7.811 ms and isolated the maxconn-free version of the prior best x3 combo, so it was gated with x10. |
| `combo_libtorch_grad_zero_dprep3_x10_20260521` | rejected-x10 | Same stack as `combo_libtorch_grad_zero_dprep3_x3_20260521` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_dprep3_x10_20260521`, x10 avg 2497.490 ms | Longer gate rejected the short-run candidate: removing maxconn1 reduced late drift versus the three-way combo but still finished 4.357 ms slower than stable x10. Default binary restored to `cbcf72b7...`. |
| `combo_libtorch_dresidual_zero_dprep3_x3_20260521` | rejected-after-x10 | `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `LLMK_SM120_DPREP_WARPS=3` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_dprep3_x3_20260521`, x3 avg 2488.597 ms | Short run beat stable x10 by 4.536 ms and combined the dresidual-zero near-miss with dprep=3, so it was gated with x10. |
| `combo_libtorch_dresidual_zero_dprep3_x10_20260521` | rejected-near-miss | Same stack as `combo_libtorch_dresidual_zero_dprep3_x3_20260521` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_dprep3_x10_20260521`, x10 avg 2494.706 ms | Closest recent x10 composition, but still 1.573 ms slower than stable x10. Do not promote a noise-level non-win; default binary restored to `cbcf72b7...`. |
| `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x3_20260521` | rejected-after-x10 | `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024` | `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x3_20260521`, x3 avg 2486.084 ms | Short run beat stable x10 by 7.049 ms by replacing the default gradient memset with the native CUDA zero kernel in the prior closest Torch+dprep composition, so it was gated with x10. |
| `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x10_20260521` | rejected-near-miss | Same stack as `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x3_20260521` | `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x10_20260521`, x10 avg 2494.320 ms | The CUDA gradient-zero route improved the prior Torch+dprep x10 by 0.387 ms, but it is still 1.187 ms slower than stable x10. Do not promote; default binary restored to `66c0932a...`. |
| `combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x3_20260521` | rejected-after-x10 | `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x3_20260521`, x3 avg 2485.470 ms | Best x3 in this composition pass, beating stable x10 by 7.663 ms, so it was gated with x10. |
| `combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x10_20260521` | rejected-near-miss | Same stack as `combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x3_20260521` | `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x10_20260521`, x10 avg 2495.009 ms | Short-run win did not survive the stability gate; it is 1.876 ms slower than stable x10. Do not promote; default binary restored to `17970026...`. |
| `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x3_20260521` | rejected-after-x10 | `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` | `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x3_20260521`, x3 avg 2484.400 ms | Best x3 in this combination pass, beating stable x10 by 8.733 ms, so it was gated with x10. |
| `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x10_20260521` | rejected-near-miss | Same stack as `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x3_20260521` | `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x10_20260521`, x10 avg 2493.735 ms | Closest current composed x10 result, but still 0.602 ms slower than stable x10. Do not promote; default binary restored to `d40f2ece...`. |
| `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x3_20260521` | rejected-after-x10 | `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `LLMK_SM120_DPREP_WARPS=4`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` | `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x3_20260521`, x3 avg 2486.240 ms | Source-default dprep=4 variant still beat stable x10 by 6.893 ms, so it was gated to test whether it had less late drift than dprep=3. |
| `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x10_20260521` | rejected-near-miss | Same stack as `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x3_20260521` | `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x10_20260521`, x10 avg 2493.970 ms | Dprep=4 did not improve stability versus dprep=3 and is 0.837 ms slower than stable x10. Do not promote; default binary restored to `aa904a1a...`. |
| `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x3_20260521` | rejected-after-x10 | `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=512`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` | `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x3_20260521`, x3 avg 2485.324 ms | 512-thread memory block variant beat stable x10 by 7.809 ms on x3, so it was gated with x10 before promotion. |
| `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x10_20260521` | rejected-near-miss | Same stack as `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x3_20260521` | `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x10_20260521`, x10 avg 2494.177 ms | Block512 was slower than the block1024 four-way x10 and remains 1.044 ms slower than stable x10. Do not promote; default binary restored to `b350a662...`. |
| `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x3_20260521` | rejected-after-x10 | `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=256`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` | `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x3_20260521`, x3 avg 2485.912 ms | 256-thread memory block variant beat stable x10 by 7.221 ms on x3, so it was gated with x10 before promotion. |
| `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x10_20260521` | rejected | Same stack as `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x3_20260521` | `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x10_20260521`, x10 avg 2499.345 ms | Block256 regressed versus both block512 and block1024, with a step-6 tail spike; it is 6.212 ms slower than stable x10. Do not promote; default binary restored to `4380d98c...`. |
| `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x3_20260521` | promoted-after-x10 | `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` | `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x3_20260521`, x3 avg 2481.166 ms | Best short-run composed stack in this pass, so it was gated with x10 and then confirmed. |
| `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x10_20260521` | promoted-confirmed | Same stack as maxconn1 x3 | `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x10_20260521`, x10 avg 2490.940 ms | First x10 gate beat stable baseline by 2.193 ms. |
| `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_confirm_x10_20260521` | promoted-confirmed | Same stack as maxconn1 x3 | `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_confirm_x10_20260521`, x10 avg 2490.432 ms | Confirmation x10 beat stable baseline by 2.701 ms, so this stack was promoted into SM120 defaults and `train-sm120.sh`. |
| `direct_train_sm120_promoted_fast_default_x10_20260521` | promoted-direct | Promoted SM120 defaults plus `CUDA_DEVICE_MAX_CONNECTIONS=1` in `train-sm120.sh` | `log124M/5090_S`, x10 avg 2489.062 ms | Direct user-facing script now beats the prior stable baseline by 4.071 ms and confirms the promoted stack is not harness-only. |
| `direct_train_sm120_promoted_slow_dmon_x10_20260521` | regression reference | Same promoted SM120 defaults and direct `./train-sm120.sh` path, with concurrent `nvidia-smi dmon` telemetry | `scratch/train-sm120-promoted-slow-dmon-20260521.log`, x10 avg 2610.981 ms; telemetry `scratch/dmon-train-sm120-promoted-slow-20260521.log` | Same binary/settings now run slower while startup still confirms promoted backends and telemetry shows full SM utilization, ~573-575 W, pclk ~2707-2782 MHz, and no violation flags. Treat as current runtime drift, not a kernel-selection change. |
| `promoted_bias_wide1024_x3_20260521` | rejected | Promoted SM120 fast default plus `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024` | `scratch/sm120_rounds/codex_sm120_promoted_bias_wide1024_x3_20260521`, x3 avg 2608.036 ms | Correctness passed, but adding the wide-bias block to the promoted stack regressed by 118.973 ms versus the promoted direct x10 average and by 114.903 ms versus the prior stable x10 baseline. Do not x10-gate or promote. |
| `promoted_no_maxconn_x3_20260521` | rejected-after-x10 | Promoted SM120 fast default with `CUDA_DEVICE_MAX_CONNECTIONS` unset | `scratch/sm120_rounds/codex_sm120_promoted_no_maxconn_x3_20260521`, x3 avg 2597.961 ms | Short run beat the current slow maxconn1 direct band, so it was x10-gated before changing the script default. |
| `promoted_no_maxconn_x10_20260521` | rejected | Same as no-maxconn x3 | `scratch/sm120_rounds/codex_sm120_promoted_no_maxconn_x10_20260521`, x10 avg 2631.929 ms | Longer gate rejected the no-maxconn hypothesis: it is slower than the current maxconn1 telemetry reference and much slower than the promoted direct proof. Keep `train-sm120.sh` exporting `CUDA_DEVICE_MAX_CONNECTIONS=1`. |
| `promoted_maxconn2_x3_20260521` | rejected | Promoted SM120 fast default with `CUDA_DEVICE_MAX_CONNECTIONS=2` | `scratch/sm120_rounds/codex_sm120_promoted_maxconn2_x3_20260521`, x3 avg 2613.712 ms | Worse than both no-maxconn x3 and the current maxconn1 direct telemetry reference. Do not promote. |
| `promoted_cuda_dresidual_x3_20260521` | rejected | Promoted compile-time stack but CUDA runtime residual zeroing instead of Torch C++ | `scratch/sm120_rounds/codex_sm120_promoted_cuda_dresidual_x3_20260521`, x3 avg 2624.266 ms | Replacing the promoted Torch C++ residual clear with CUDA runtime zeroing regressed; keep Torch C++ dresidual-zero in the promoted stack. |
| `promoted_cublas_dinp_attproj_fc_x3_20260521` | rejected | Promoted SM120 fast default plus `LLMK_SM120_USE_CUBLAS_DINP_ATTPROJ` and `LLMK_SM120_USE_CUBLAS_DINP_FC` | `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_attproj_fc_x3_20260521`, x3 avg 2618.905 ms | Correctness passed, but revisiting direct-cuBLAS dInput on top of the promoted stack regressed versus both the current slow maxconn1 reference and the promoted direct proof. Do not promote. |
| `promoted_classifier_exp2_x3_20260521` | rejected-current-band-win | Promoted SM120 fast default plus `LLMK_SM120_CLASSIFIER_EXP2` | `scratch/sm120_rounds/codex_sm120_promoted_classifier_exp2_x3_20260521`, x3 avg 2580.789 ms | Correctness passed and it beat the current slow maxconn1 telemetry reference by 30.192 ms, but it is still 91.727 ms slower than the promoted direct proof. Do not promote or x10-gate. |
| `promoted_tk_dgelu_x3_20260521` | rejected-current-band-win | Promoted SM120 fast default plus `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` | `scratch/sm120_rounds/codex_sm120_promoted_tk_dgelu_x3_20260521`, x3 avg 2502.759 ms | Correctness passed and this is much faster than the current slow band, but it is still 13.696 ms slower than the promoted direct proof and 9.626 ms slower than stable x10. Do not promote or x10-gate. |
| `promoted_tk_dgelu_classifier_exp2_x3_20260521` | rejected | Promoted SM120 fast default plus `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` and `LLMK_SM120_CLASSIFIER_EXP2` | `scratch/sm120_rounds/codex_sm120_promoted_tk_dgelu_classifier_exp2_x3_20260521`, x3 avg 2609.867 ms | Correctness passed, but combining the two current-band add-ons destroyed the TK dGELU gain and nearly returned to the slow maxconn1 band. Do not promote or x10-gate. |
| `promoted_precompute_grad_scale_x3_20260521` | rejected-current-band-win | Promoted SM120 fast default plus `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` | `scratch/sm120_rounds/codex_sm120_promoted_precompute_grad_scale_x3_20260521`, x3 avg 2604.345 ms | Correctness passed and it beat the current slow maxconn1 telemetry reference by 6.636 ms, but it is still 115.283 ms slower than the promoted direct proof and 111.212 ms slower than stable x10. Do not promote or x10-gate. |
| `promoted_libtorch_grad_zero_x3_20260521` | rejected-current-band-win | Promoted SM120 fast default but LibTorch grad-zero replaces the promoted CUDA-kernel grad-zero route | `scratch/sm120_rounds/codex_sm120_promoted_libtorch_grad_zero_x3_20260521`, x3 avg 2593.987 ms | Correctness passed and it beat the current slow maxconn1 telemetry reference by 16.994 ms, but it is still 104.925 ms slower than the promoted direct proof and 100.854 ms slower than stable x10. Do not promote or x10-gate. |
| `promoted_libtorch_grad_zero_tk_dgelu_x3_20260521` | rejected | Promoted SM120 fast default plus LibTorch grad-zero and `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` | `scratch/sm120_rounds/codex_sm120_promoted_libtorch_grad_zero_tk_dgelu_x3_20260521`, x3 avg 2607.654 ms | Correctness passed, but combining the two current-band add-ons regressed versus both add-ons alone and remains 118.592 ms slower than the promoted direct proof. Do not promote or x10-gate. |
| `promoted_cublas_dinp_tk_dgelu_x3_20260521` | rejected-after-x10 | Promoted SM120 fast default plus `LLMK_SM120_USE_CUBLAS_DINP_ATTPROJ`, `LLMK_SM120_USE_CUBLAS_DINP_FC`, and `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` | `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_tk_dgelu_x3_20260521`, x3 avg 2491.288 ms | Correctness passed and the short run beat stable x10 by 1.845 ms, so it was gated with x10 before any promotion. |
| `promoted_cublas_dinp_tk_dgelu_x10_20260521` | rejected | Same stack as `promoted_cublas_dinp_tk_dgelu_x3_20260521` | `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_tk_dgelu_x10_20260521`, x10 avg 2508.456 ms | Longer gate rejected the short-run candidate: it is 19.394 ms slower than the promoted direct proof and 15.323 ms slower than stable x10. Promoted-default binaries were rebuilt afterward and the restored smoke suite passed. |
| `promoted_memory_store_cg_x3_20260521` | rejected | Promoted SM120 fast default plus `LLMK_SM120_MEMORY_STORE_POLICY=2` for the native memory zero/copy wrappers | `scratch/sm120_rounds/codex_sm120_promoted_memory_store_cg_x3_20260521`, x3 avg 2592.754 ms | Correctness passed, but changing the promoted CUDA-kernel grad-zero store path to cache-global stores regressed by 103.692 ms versus promoted direct. Do not x10-gate or promote. |
| `promoted_memory_store_default_x3_20260521` | rejected | Promoted SM120 fast default plus `LLMK_SM120_MEMORY_STORE_POLICY=0` for the native memory zero/copy wrappers | `scratch/sm120_rounds/codex_sm120_promoted_memory_store_default_x3_20260521`, x3 avg 2593.304 ms | Correctness passed, but regular stores are also far slower than the promoted default streaming-store policy. Keep `LLMK_SM120_MEMORY_STORE_POLICY=1`. |
| `promoted_global_norm_block256_x3_20260521` | rejected-current-band-win | Promoted SM120 fast default plus `LLMK_SM120_GLOBAL_NORM_BLOCK_SIZE=256` | `scratch/sm120_rounds/codex_sm120_promoted_global_norm_block256_x3_20260521`, x3 avg 2590.486 ms | Correctness passed and it beat the current slow maxconn1 telemetry reference by 20.495 ms, but it is still 101.423 ms slower than the promoted direct proof and 97.353 ms slower than stable x10. Do not x10-gate or promote. |
| `promoted_cublaslt_plan_cache_x3_20260521` | rejected-current-band-win | Promoted SM120 fast default plus `LLMK_SM120_CACHE_CUBLASLT_PLANS` | `scratch/sm120_rounds/codex_sm120_promoted_cublaslt_plan_cache_x3_20260521`, x3 avg 2597.241 ms | Correctness passed and it beat the current slow maxconn1 telemetry reference by 13.740 ms, but it is still 108.178 ms slower than the promoted direct proof and 104.108 ms slower than stable x10. Do not x10-gate or promote. |
| `promoted_cublaslt_ws256_heur16_x3_20260521` | rejected-current-band-win | Promoted SM120 fast default plus `LLMK_SM120_CUBLASLT_WORKSPACE_MB=256` and `LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS=16` | `scratch/sm120_rounds/codex_sm120_promoted_cublaslt_ws256_heur16_x3_20260521`, x3 avg 2603.521 ms | Correctness passed and it beat the current slow maxconn1 telemetry reference by 7.460 ms, but it is still 114.459 ms slower than the promoted direct proof and 110.388 ms slower than stable x10. Do not x10-gate or promote. |
| `promoted_cublas_dinp_qkv_x3_20260521` | rejected-current-band-win | Promoted SM120 fast default plus `LLMK_SM120_USE_CUBLAS_DINP_QKV` | `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_qkv_x3_20260521`, x3 avg 2607.288 ms | Correctness passed and it beat the current slow maxconn1 telemetry reference by 3.693 ms, but it is still 118.226 ms slower than the promoted direct proof and 114.155 ms slower than stable x10. Do not x10-gate or promote. |
| `promoted_cublas_dinp_fcproj_x3_20260521` | rejected-near-miss | Promoted SM120 fast default plus `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` | `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_fcproj_x3_20260521`, x3 avg 2493.911 ms | Correctness passed and this is much faster than the current slow maxconn1 telemetry reference, but it is still 4.849 ms slower than the promoted direct proof and 0.778 ms slower than stable x10. Do not x10-gate or promote. |
| `promoted_cublas_dinp_fcproj_x10_20260522` | rejected-after-confirm | Promoted SM120 fast default plus `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` | `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_fcproj_x10_20260522`, x10 avg 2490.019 ms | This near-miss was x10-gated because it was the closest remaining promoted-stack dInput selector. The first x10 beat stable x10 by 3.114 ms and the same-session default by 43.465 ms, but it was still 0.957 ms slower than the promoted direct proof and required confirmation before promotion. |
| `promoted_default_same_session_x10_20260522` | regression-control | Promoted SM120 fast default without extra candidate flags | `scratch/sm120_rounds/codex_sm120_promoted_default_same_session_x10_20260522`, x10 avg 2533.484 ms | Same-session control for the fcproj gate. The promoted stack showed late-step spikes under the current runtime band, explaining why `train-sm120.sh` can be slower now without a kernel-stack source change. |
| `promoted_cublas_dinp_fcproj_confirm_x10_20260522` | rejected-unstable-confirm | Promoted SM120 fast default plus `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` | `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_fcproj_confirm_x10_20260522`, x10 avg 2548.056 ms | Confirmation failed with late-run spikes. The candidate is 58.994 ms slower than promoted direct proof, 54.923 ms slower than stable x10, and 14.573 ms slower than the same-session promoted default. Do not promote. |
| `promoted_async_grad_norm_copy_x3_20260522` | rejected | Promoted SM120 fast default plus `LLMK_SM120_ASYNC_GRAD_NORM_COPY` | `scratch/sm120_rounds/codex_sm120_promoted_async_grad_norm_copy_x3_20260522`, x3 avg 2508.091 ms | Correctness passed, but the stream-scoped async grad-norm scalar copy is 19.029 ms slower than promoted direct proof and 14.958 ms slower than stable x10. Do not x10-gate or promote. |
| `promoted_device_grad_scale_adamw_x3_20260522` | rejected | Promoted SM120 fast default plus `LLMK_SM120_DEVICE_GRAD_SCALE_ADAMW` | `scratch/sm120_rounds/codex_sm120_promoted_device_grad_scale_adamw_x3_20260522`, x3 avg 2528.446 ms | Correctness passed and startup confirmed `grad_scale_backend = device AdamW scalar`, but this is 39.384 ms slower than promoted direct proof and 35.313 ms slower than stable x10. Do not x10-gate or promote. |
| `promoted_maxconn4_x3_20260522` | rejected | Promoted SM120 fast default with `CUDA_DEVICE_MAX_CONNECTIONS=4` | `scratch/sm120_rounds/codex_sm120_promoted_maxconn4_x3_20260522`, x3 avg 2531.798 ms | Correctness passed and this runtime scheduling setting beat maxconn2, but it is 42.735 ms slower than promoted direct proof and 38.665 ms slower than stable x10. Keep maxconn1. |
| `promoted_maxconn8_x3_20260522` | rejected-after-x10 | Promoted SM120 fast default with `CUDA_DEVICE_MAX_CONNECTIONS=8` | `scratch/sm120_rounds/codex_sm120_promoted_maxconn8_x3_20260522`, x3 avg 2483.451 ms | Correctness passed and the short run beat promoted direct proof by 5.611 ms, so it was x10-gated before any script change. |
| `promoted_maxconn8_x10_20260522` | rejected | Same stack as `promoted_maxconn8_x3_20260522` | `scratch/sm120_rounds/codex_sm120_promoted_maxconn8_x10_20260522`, x10 avg 2509.947 ms | Longer gate rejected maxconn8: late spikes made it 20.885 ms slower than promoted direct proof and 16.814 ms slower than stable x10. Keep maxconn1. |
| `promoted_maxconn16_x3_20260522` | rejected-after-x10 | Promoted SM120 fast default with `CUDA_DEVICE_MAX_CONNECTIONS=16` | `scratch/sm120_rounds/codex_sm120_promoted_maxconn16_x3_20260522`, x3 avg 2485.113 ms | Correctness passed and the short run beat promoted direct proof by 3.949 ms, so it was x10-gated before any script change. |
| `promoted_maxconn16_x10_20260522` | rejected | Same stack as `promoted_maxconn16_x3_20260522` | `scratch/sm120_rounds/codex_sm120_promoted_maxconn16_x10_20260522`, x10 avg 2525.956 ms | Longer gate rejected maxconn16: early and late spikes made it 36.894 ms slower than promoted direct proof and 32.823 ms slower than stable x10. Keep maxconn1. |
| `promoted_default_post_sched_x10_20260522` | regression-control | Promoted SM120 fast default with selected `CUDA_DEVICE_MAX_CONNECTIONS=1` after the scheduler sweep | `scratch/sm120_rounds/codex_sm120_promoted_default_post_sched_x10_20260522`, x10 avg 2543.390 ms | Same selected stack is currently slower than earlier proof, with spikes on steps 2-3 and 9-10. This explains the slower script band without attributing it to a promoted candidate left enabled. |
| `promoted_attn_fwd16_x3_20260522` | rejected | Promoted SM120 fast default plus `LLMK_SM120_ATTN_FWD_BLOCK=16` | `scratch/sm120_rounds/codex_sm120_promoted_attn_fwd16_x3_20260522`, x3 avg 2531.237 ms; attention fwd/bwd 1055.652/2747.087 us | Correctness passed, but the 16-row forward tile worsened the selected packed-QKV attention forward benchmark and remained 42.175 ms slower than promoted direct proof. Do not x10-gate or promote. |
| `promoted_attn_bwd32_x3_20260522` | rejected-current-band-win | Promoted SM120 fast default plus `LLMK_SM120_ATTN_BWD_BLOCK=32` | `scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522`, x3 avg 2502.501 ms; attention fwd/bwd 830.444/3074.490 us | Correctness passed and it beat the latest slow selected-stack control by 40.889 ms, but the focused attention benchmark worsened and the trainer still trails promoted direct proof by 13.439 ms and stable x10 by 9.368 ms. Do not x10-gate or promote. |
| `cublaslt_epilogue_probe_20260522` | benchmark-rejected | Focused cuBLASLt fused-epilogue heuristic-index probe for `fc fwd+GeLU` and `fcproj dInp+dGeLU` | `scratch/sm120_rounds/codex_sm120_cublaslt_epilogue_probe_20260522/bench_sm120_cublaslt_epilogue_algos.log`; no trainer run | Probe does not justify a trainer candidate: `fc` default index 0 is best, and the apparent `fcproj` index-1 edge vanished when default index 0 was retested faster. |
| `promoted_default_after_epilogue_probe_x3_20260522` | regression-control | Restored promoted SM120 fast default after the epilogue probe, no extra candidate flags | `scratch/sm120_rounds/codex_sm120_promoted_default_after_epilogue_probe_x3_20260522`, x3 avg 2594.888 ms | Correctness passed, but the selected stack is still in a slow runtime band: 105.826 ms slower than promoted direct proof and 101.755 ms slower than stable x10. |
| `direct_train_sm120_current_no_rebuild_x10_20260522` | regression-control | Exact `./train-sm120.sh` path with current promoted-default binary and `CUDA_DEVICE_MAX_CONNECTIONS=1` | `scratch/sm120_rounds/codex_sm120_direct_train_sm120_current_no_rebuild_x10_20260522/train-sm120.log`, x10 avg 2541.633 ms | Direct script confirms the user-facing path is currently slower while selected backends remain active: CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad-scale, and BF16 on RTX 5090. This is 52.571 ms slower than promoted direct proof and 48.500 ms slower than stable x10, so it is not a new promotion baseline. |
| `promoted_cublaslt_heur1_x3_20260522` | rejected | Promoted SM120 fast default plus `LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS=1` | `scratch/sm120_rounds/codex_sm120_promoted_cublaslt_heur1_x3_20260522`, x3 avg 2552.864 ms; benchmarks captured | Correctness and focused benchmarks passed, but forcing one cuBLASLt heuristic is 63.802 ms slower than promoted direct proof, 59.731 ms slower than stable x10, and 11.232 ms slower than the current direct-script regression control. Do not x10-gate or promote. |
| `promoted_disable_cublas_bwd_x3_20260522` | rejected-current-band-win | Promoted SM120 fast default plus `LLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM` | `scratch/sm120_rounds/codex_sm120_promoted_disable_cublas_bwd_x3_20260522`, x3 avg 2521.217 ms; benchmarks captured | Correctness passed and this improved the current direct-script regression control by 20.416 ms, but it is still 32.154 ms slower than promoted direct proof and 28.084 ms slower than stable x10. Do not x10-gate or promote. |
| `cublas_dinp_fcproj_maxconn8_x3_20260522` | rejected | Promoted SM120 fast default plus `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` and `CUDA_DEVICE_MAX_CONNECTIONS=8` | `scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_maxconn8_x3_20260522`, x3 avg 2678.119 ms; benchmarks captured | Correctness passed, but composing the unstable fcproj direct-cuBLAS selector with maxconn8 severely regressed training: 189.057 ms slower than promoted direct proof, 184.986 ms slower than stable x10, and 194.668 ms slower than maxconn8 alone. Do not x10-gate or promote. |
| `promoted_disable_backward_stream_sync_x3_20260522` | rejected-current-band-partial | Promoted SM120 fast default plus `LLMK_SM120_DISABLE_BACKWARD_STREAM_SYNC` | `scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522`, x3 avg 2535.141 ms; benchmarks captured | Correctness passed and the older device-wide backward sync was 6.492 ms faster than the current direct-script regression control, but it is still 46.078 ms slower than promoted direct proof and 42.008 ms slower than stable x10. Keep the promoted stream-scoped sync. |
| `antigravity_binary_probe_x3_20260522` | diagnostic-rejected | Separate historical `train_gpt2cu_antigravity` binary, not a source-promotable stack | `log124M/5090_S_antigravity_probe_x3_20260522`, x3 avg 3312.640 ms | Diagnostic run rejects the hypothesis that this older binary explains the `new-goal.md` 2469.843 ms target. It is 842.797 ms slower than that target, 823.578 ms slower than promoted direct proof, and 771.007 ms slower than the current direct-script regression control. Do not promote; keep as reference only. |
| `current_selected_runtime_telemetry_x3_20260522` | recovered-band-control | Current selected `train_gpt2cu` stack with `CUDA_DEVICE_MAX_CONNECTIONS=1` and dmon telemetry | `log124M/5090_S_runtime_telemetry_probe_x3_20260522`, x3 avg 2479.394 ms | The same selected CUDA/Torch stack returned to a fast band: 62.239 ms faster than the degraded direct-script control and 9.668 ms faster than promoted direct proof, but still 9.551 ms slower than the `new-goal.md` first-three target. Telemetry under load showed full utilization, high clocks, and no power or thermal violation flags. |
| `direct_train_sm120_recovered_x10_20260522` | recovered-band-control | Exact `./train-sm120.sh` path with current selected binary after telemetry probe | `log124M/5090_S`, x10 avg 2489.794 ms | Exact script no longer reproduces the 2541.633 ms slow band. This x10 is 51.839 ms faster than the degraded direct-script control and 3.339 ms faster than stable x10, but 0.731 ms slower than promoted direct proof and 19.951 ms slower than the `new-goal.md` first-three target. |
| `promoted_precompute_grad_scale_recovered_x3_20260522` | rejected-near-current | Promoted SM120 fast default plus `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` retested in the recovered runtime band | `scratch/sm120_rounds/codex_sm120_promoted_precompute_grad_scale_recovered_x3_20260522`, x3 avg 2480.090 ms | Correctness passed and the precomputed device AdamW scalar route stayed in the fast band, but it is 0.696 ms slower than the same-band selected-stack x3 and 10.247 ms slower than the `new-goal.md` target. Restored promoted default after the run. |
| `zero_stage0_probe_x3_20260522` | rejected-near-current | Current selected binary with single-GPU `-z 0` instead of script default `-z 1` | `log124M/5090_S_zero0_probe_x3_20260522`, x3 avg 2480.059 ms | Disabling ZeRO-1 on the one-process trainer is correctness-consistent by loss/norm trace, but it is 0.665 ms slower than the same-band selected-stack x3 and 10.216 ms slower than the `new-goal.md` target. Keep `train-sm120.sh` on `-z 1`. |
| `precompute_grad_scale_zero0_x3_20260522` | rejected | Promoted SM120 fast default plus `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` and `TRAIN_ZERO_STAGE=0` | `scratch/sm120_rounds/codex_sm120_precompute_grad_scale_zero0_x3_20260522`, x3 avg 2488.396 ms | Correctness passed, but combining the two recovered-band near misses regressed by 9.002 ms versus the selected-stack x3 and remains 18.553 ms slower than the `new-goal.md` target. Keep host-scalar AdamW grad scale and `-z 1`. |
| `direct_train_sm120_fresh_slow_control_x10_20260522` | regression-control | Exact `./train-sm120.sh` with active restored promoted-default binary `a55f2b0e...` | `scratch/sm120_rounds/codex_sm120_direct_train_sm120_fresh_slow_control_x10_20260522/train-sm120.log`, x10 avg 2575.607 ms | Same selected CUDA/Torch backend mix is active, but the direct script is back in a slow runtime band; this is not evidence of a rejected kernel component being enabled. |
| `fresh_rebuild_selected_control_x3_20260522` | control | Fresh harness rebuild of the selected promoted stack | `scratch/sm120_rounds/codex_sm120_fresh_rebuild_selected_control_x3_20260522`, x3 avg 2491.280 ms | Fresh rebuild recovered most of the severe slow band but remained 21.437 ms slower than the `new-goal.md` target. |
| `precompute_grad_scale_maxconn8_x3_20260522` | rejected | `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` plus `CUDA_DEVICE_MAX_CONNECTIONS=8` | `scratch/sm120_rounds/codex_sm120_precompute_grad_scale_maxconn8_x3_20260522`, x3 avg 2877.633 ms | Correctness passed, but combining the precomputed AdamW scalar route with maxconn8 caused severe warmup and memory-pressure regression. Do not x10-gate or promote. |
| `user_observed_train_sm120_fast_first5_20260522` | external-observation-needs-reproduction | Exact `./train-sm120.sh` pasted by user with selected backend mix | conversation excerpt, first3 avg 2461.450 ms; first5 avg 2462.236 ms | Promising: first3 beats `new-goal.md` by 8.393 ms, but Codex reruns on the active binary did not reproduce it, so this is not goal-complete evidence yet. |
| `direct_train_sm120_a9f_verify2_x10_20260522` | reproduction-miss | Exact `./train-sm120.sh` with active binary `a9f1277e...` | `scratch/sm120_rounds/direct_train_sm120_a9f_verify2_x10_20260522.log`, x10 avg 2511.285 ms | Same selected backend names and active binary did not reproduce the user's fast band; first3 averaged 2503.977 ms. |
| `direct_train_sm120_telemetry_verify3_x10_20260522` | recovered-band-reproduction-miss | Exact `./train-sm120.sh` with active binary `a9f1277e...` and `nvidia-smi dmon` telemetry | `scratch/sm120_rounds/codex_sm120_direct_train_telemetry_verify3_20260522/train-sm120.log`, x10 avg 2485.844 ms; first3 avg 2479.820 ms | Runtime recovered versus the slow 2511 ms rerun, with full SM utilization and no power/thermal violation flags, but still missed the user's 2461 ms first-three band by 18.370 ms. |
| `direct_train_sm120_a9f_warm_verify4_x10_20260522` | recovered-band-reproduction-miss | Exact `./train-sm120.sh` warm follow-up with active binary `a9f1277e...` | `scratch/sm120_rounds/direct_train_sm120_a9f_warm_verify4_x10_20260522.log`, x10 avg 2489.568 ms; first3 avg 2483.410 ms | Warm follow-up stayed in the recovered band but did not reproduce the user's fast band or beat `new-goal.md`; no kernel promotion follows from it. |
| `direct_train_sm120_a9f_verify5_x10_20260522` | recovered-band-reproduction-miss | Exact `./train-sm120.sh` on an idle GPU with active binary `a9f1277e...` | `scratch/sm120_rounds/direct_train_sm120_a9f_verify5_x10_20260522/train-sm120.log`, x10 avg 2484.309 ms; first3 avg 2478.143 ms | Best Codex reproduction so far and faster than the promoted x10 proof, but still 8.300 ms slower than `new-goal.md` first-three target and 16.693 ms slower than the user-pasted first-three band. |
| `direct_train_sm120_a9f_maxconn8_verify6_x10_20260522` | rejected-recovered-band-runtime | Exact `./train-sm120.sh` with active binary `a9f1277e...` and `CUDA_DEVICE_MAX_CONNECTIONS=8` | `scratch/sm120_rounds/direct_train_sm120_a9f_maxconn8_verify6_x10_20260522/train-sm120.log`, x10 avg 2485.662 ms; first3 avg 2479.587 ms | Maxconn8 remained slower than the same-band maxconn1 direct rerun by 1.353 ms x10 and 1.443 ms first3. Keep `train-sm120.sh` on maxconn1. |
| `direct_train_sm120_a9f_maxconn16_verify7_x10_20260522` | rejected-recovered-band-runtime | Exact `./train-sm120.sh` with active binary `a9f1277e...` and `CUDA_DEVICE_MAX_CONNECTIONS=16` | `scratch/sm120_rounds/direct_train_sm120_a9f_maxconn16_verify7_x10_20260522/train-sm120.log`, x10 avg 2485.517 ms; first3 avg 2478.910 ms | Maxconn16 was marginally faster than maxconn8, but still slower than the same-band maxconn1 direct rerun by 1.208 ms x10 and 0.767 ms first3. Keep maxconn1. |
| `runtime_grad_zero_recovered_x3_20260522` | rejected | Promoted SM120 fast default but CUDA runtime grad-zero replaces CUDA-kernel grad-zero | `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_recovered_x3_20260522`, trainer avg 2478.070 ms; visible first3 avg 2479.853 ms | Correctness passed, but the visible first-three average is slower than the selected same-band maxconn1 run by 1.710 ms and slower than `new-goal.md` by 10.010 ms. Restored selected CUDA-kernel grad-zero build. |
| `direct_train_sm120_9fe_restore_verify8_x10_20260522` | recovered-band-control | Exact `./train-sm120.sh` after restoring selected CUDA-kernel grad-zero build, active binary `9fe90db1...` | `scratch/sm120_rounds/direct_train_sm120_9fe_restore_verify8_x10_20260522/train-sm120.log`, x10 avg 2485.751 ms; first3 avg 2478.147 ms | Rebaseline confirms the restored selected stack is still in the recovered band and ties the prior best first-three within 0.004 ms, but it does not beat `new-goal.md`; no promotion change. |
| `direct_train_sm120_9fe_user_rerun_verify9_x10_20260522` | recovered-band-reproduction-miss | Exact `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh` on idle GPU with active binary `9fe90db1...` | `scratch/sm120_rounds/direct_train_sm120_9fe_user_rerun_verify9_x10_20260522/train-sm120.log`, x10 avg 2485.469 ms; first3 avg 2479.387 ms | User-requested rerun did not reproduce the pasted 2461 ms first-three band. It is 9.544 ms slower than `new-goal.md` first-three and 17.937 ms slower than the user-observed first-three band, so no promotion or goal completion follows. |
| `direct_train_sm120_aa2_user_rerun_verify10_x10_20260522` | recovered-band-reproduction-miss | Exact `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh` on idle GPU with restored selected binary `aa2d2499...` | `scratch/sm120_rounds/direct_train_sm120_aa2_user_rerun_verify10_x10_20260522/train-sm120.log`, x10 avg 2490.359 ms; first3 avg 2481.190 ms | User-suggested rerun after reporting the prior GPU may have been busy still did not reproduce the pasted 2461 ms first-three band. It is 11.347 ms slower than `new-goal.md` first-three and 19.740 ms slower than the user-observed first-three band, so no promotion or goal completion follows. |
| `direct_train_sm120_0452_user_followup_verify11_x10_20260522` | recovered-band-reproduction-miss | Exact `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh` on idle GPU with active selected binary `0452da63...` | `scratch/sm120_rounds/direct_train_sm120_0452_user_followup_verify11_x10_20260522/train-sm120.log`, x10 avg 2489.367 ms; first3 avg 2482.793 ms | Follow-up rerun after the user's faster pasted result again missed the 2461 ms band. It is 12.950 ms slower than `new-goal.md` first-three and 21.343 ms slower than the user-observed first-three band, so no promotion or goal completion follows. |
| `direct_train_sm120_0452_exact_newgoal_x3_20260522` | exact-command-control | Exact `new-goal.md` three-step command on idle GPU with active selected binary `0452da63...` | `scratch/sm120_rounds/direct_train_sm120_0452_exact_newgoal_x3_20260522/train-sm120-x3.log`, trainer avg 2481.710 ms; visible first3 avg 2483.187 ms | The exact `-x 3` command did not reproduce `new-goal.md` or the user-observed 2461 ms band. It remains 13.344 ms slower than `new-goal.md` first-three and 21.737 ms slower than the user-observed first-three band. |
| `promoted_disable_cuda_profiler_recovered_x3_20260522` | rejected | Selected promoted stack plus `LLMK_DISABLE_CUDA_PROFILER`, retested in the recovered band | `scratch/sm120_rounds/codex_sm120_promoted_disable_cuda_profiler_recovered_x3_20260522`, trainer avg 2482.823 ms; visible first3 avg 2484.143 ms | Correctness passed, but disabling profiler calls is slightly slower than the exact selected-stack x3 control and remains 14.300 ms slower than `new-goal.md`. Restored selected binary and reran all nine focused smokes. |
| `dprep2_recovered_x3_20260522` | rejected | Selected promoted stack with `LLMK_SM120_DPREP_WARPS=2` replacing dprep=3; cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, memory block 1024, LayerNorm bwd1, maxconn1 | `scratch/sm120_rounds/codex_sm120_dprep2_recovered_x3_20260522`, trainer avg 2485.089 ms; visible first3 avg 2486.753 ms; attention 787.510/2737.404 us | Correctness passed and focused attention was near selected timing, but trainer first3 regressed by 7.367 ms versus the latest selected-stack direct rerun and is 16.910 ms slower than `new-goal.md`. Restored selected dprep=3 build and reran all nine focused smokes. |
| `dprep1_recovered_x3_20260522` | rejected | Selected promoted stack with `LLMK_SM120_DPREP_WARPS=1` replacing dprep=3; cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, memory block 1024, LayerNorm bwd1, maxconn1 | `scratch/sm120_rounds/codex_sm120_dprep1_recovered_x3_20260522`, trainer avg 2495.000 ms; visible first3 avg 2497.420 ms; attention 784.284/2857.293 us | Correctness passed, but the lower dprep warp count worsened attention backward and trainer first3: 18.033 ms slower than the latest selected-stack direct rerun and 27.577 ms slower than `new-goal.md`. Restored selected dprep=3 build and reran all nine focused smokes. |
| `attn_fwd64_recovered_x3_20260522` | rejected | Selected promoted stack with `LLMK_SM120_ATTN_FWD_BLOCK=64` replacing the default attention forward tile; selected bwd tile, dprep=3, cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, memory block 1024, LayerNorm bwd1, maxconn1 | `scratch/sm120_rounds/codex_sm120_attn_fwd64_recovered_x3_20260522`, trainer avg 2712.468 ms; visible first3 avg 2714.013 ms; attention 3223.705/2742.139 us | Correctness passed, but the larger forward tile catastrophically worsened attention forward and trainer speed: first3 is 234.627 ms slower than the latest selected-stack direct rerun and 244.170 ms slower than `new-goal.md`. Restored selected attention tile build and reran all nine focused smokes. |
| `attn_bwd64_recovered_x3_20260522` | rejected | Selected promoted stack with `LLMK_SM120_ATTN_BWD_BLOCK=64` replacing the default attention backward tile; selected fwd tile, dprep=3, cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, memory block 1024, LayerNorm bwd1, maxconn1 | `scratch/sm120_rounds/codex_sm120_attn_bwd64_recovered_x3_20260522`, trainer avg 3901.719 ms; visible first3 avg 3912.080 ms; attention 787.890/18050.102 us | Correctness passed, but the larger backward tile catastrophically worsened attention backward and trainer speed: first3 is 1430.890 ms slower than the latest selected-stack direct rerun and 1442.237 ms slower than `new-goal.md`. Restored selected attention tile build and reran all nine focused smokes. |
| `attention_refresh_cudnn_torch_20260522` | benchmark-rejected | Refreshed native packed TK, cuDNN separated/packed, Torch separated/packed/materialized, and Torch qkv-layout attention routes | `scratch/sm120_rounds/codex_sm120_attention_refresh_20260522`; packed TK total 3519.783 us, cuDNN packed total 3613.122 us, Torch packed total 5226.573 us | Separated cuDNN/Torch rows are faster reference evidence, but trainer-compatible packed/layout routes remain slower than packed TK. No attention trainer integration or TinyStories run is justified from this refresh. |
| `runtime_grad_zero_recovered_x10_20260522` | rejected-near-current | Selected promoted stack but CUDA runtime grad-zero replaces CUDA-kernel grad-zero, x10 gate of recovered-band x3 near-miss | `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_recovered_x10_20260522`, x10 avg 2489.146 ms; first3 avg 2481.347 ms | Correctness passed and x10 was close in the recovered band, but it is 0.084 ms slower than promoted direct proof and 11.504 ms slower than `new-goal.md` first-three. Keep CUDA-kernel grad-zero selected. |
| `promoted_backward_n96_x3_20260522` | rejected | Promoted SM120 fast default plus `LLMK_SM120_BACKWARD_N96=1` | `scratch/sm120_rounds/codex_sm120_promoted_backward_n96_x3_20260522`, x3 avg 2487.329 ms; benchmarks captured | Correctness passed, but the N96 TK backward tile selector regressed training by 7.935 ms versus the recovered selected-stack x3 and remains 17.486 ms slower than the `new-goal.md` target. Keep the default `LLMK_SM120_BACKWARD_N96=0`. |
| `precompute_disable_profiler_x3_20260522` | rejected | Selected stack plus `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` and `LLMK_DISABLE_CUDA_PROFILER` | `scratch/sm120_rounds/codex_sm120_precompute_disable_profiler_x3_20260522`, trainer avg 2496.255 ms; first3 avg 2496.837 ms | Correctness passed, but composing two near-current rows regressed by 26.994 ms versus `new-goal.md`, 8.410 ms versus the latest selected direct rerun, and was slower than both component rows. Restored selected binary `a60e97a6...` and reran all nine focused smokes. |
| `zero0_disable_profiler_x3_20260522` | rejected | Selected stack with single-GPU `-z 0` plus `LLMK_DISABLE_CUDA_PROFILER` | `scratch/sm120_rounds/codex_sm120_zero0_disable_profiler_x3_20260522`, trainer avg 2496.472 ms; first3 avg 2496.810 ms | Correctness passed, but the no-ZeRO/profiler interaction regressed by 26.967 ms versus `new-goal.md`, 8.383 ms versus the latest selected direct rerun, and was slower than both component rows. Restored selected binary `a60e97a6...` and reran all nine focused smokes. |
| `direct_train_sm120_codex_rerun_after_user_fast_20260522` | reproduction-miss | Exact `./train-sm120.sh` with restored selected binary `a60e97a6...` after the user reported a faster first-five band | `scratch/sm120_rounds/direct_train_sm120_codex_rerun_after_user_fast_20260522/train-sm120.log`, x10 trainer avg 2498.704 ms; first3 avg 2489.237 ms; first5 avg 2491.620 ms | Startup confirmed the selected CUDA/Torch mix and `-z 1`, but this rerun did not reproduce the user's 2461.450 ms first-three band or beat `new-goal.md`. Generated step-10 checkpoints were removed; no promotion or goal completion follows. |
| `matmul_dbias768_x3_20260522` | rejected | Selected stack plus `LLMK_SM120_BIAS_BLOCK_SIZE=768` for matmul backward-bias reductions | `scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522`, trainer avg 2497.894 ms; first3 avg 2498.463 ms | Correctness passed and this closes the gap between default 512, rejected 256, and rejected 1024 dbias block sizes. It is 28.620 ms slower than `new-goal.md`, 9.227 ms slower than the latest selected direct rerun, 9.197 ms slower than dbias1024, and 3.033 ms slower than dbias256. Restored selected binary `a60e97a6...` and reran all nine focused smokes. |

## Trial Notes

### `default_after_combo_rebuild_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_default_after_combo_rebuild_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_default_after_combo_rebuild_x3_20260521`
- Build line contained the default SM120 `LLMK_SM120_USE_CUBLASLT_GEMM` flag only; no candidate `EXTRA_NVCC_FLAGS` and no LibTorch trainer flags.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2500.88 ms, 2495.12 ms, 2498.64 ms.
- Validator summary: `train_steps=3`, `avg_ms=2496.877`, validation OK.

### `direct_train_sm120_default_x10_20260521`

- Command: `./train-sm120.sh`
- Output directory: `log124M/5090_S`
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2498.92, 2497.08, 2499.89, 2501.18, 2503.22, 2512.56, 2510.72, 2512.98, 2514.21, 2517.06 ms.
- Average: 2507.656 ms.
- Decision: reference only. This proves the direct user script has not reproduced the historical `2469.843 ms` note after the clean default rebuild.

### `direct_train_sm120_current_rerun_x10_20260521`

- Command: `./train-sm120.sh`
- Captured stdout: `scratch/train-sm120-current-rerun-20260521.log`
- Binary before the rerun: `train_gpt2cu` sha256 `e7abfa54fb5017c53d5351d9d1d6aaa06b66fe4bbefe88d4891ace5063bde618`, size `3060032`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 3013.66, 2544.39, 2694.84, 2674.74, 2582.47, 2683.40, 2602.63, 2652.47, 2724.01, 2751.20 ms.
- Average: 2656.684 ms.
- Decision: regression reference. This is not a Torch/LibTorch route accidentally left on; the reported trainer backends are still the default CUDA runtime paths.

### `direct_train_sm120_after_profiler_rebuild_x10_20260521`

- Command: `./train-sm120.sh`
- Captured stdout: `scratch/train-sm120-direct-after-profiler-rebuild-x10-20260521.log`
- Binary after default rebuild: `train_gpt2cu` sha256 `49c6ce2870fe29aee26c4a1c47b41b5d4e34fdcfcb43e3709443c7868b23d1be`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2509.28, 2509.24, 2512.63, 2515.92, 2516.68, 2521.77, 2522.41, 2524.32, 2525.88, 2526.39 ms.
- Average: 2519.472 ms.
- Decision: regression reference. The direct script remains slower than the stable harness baseline after restoring a default binary, so this is runtime drift at the unchanged selected stack, not a leftover candidate backend.

### `direct_train_sm120_fresh_default_after_restore_x10_20260521`

- Command: `./train-sm120.sh`
- Output directory: `log124M/5090_S`
- Binary after default restore: `train_gpt2cu` sha256 `dba87b1716263f91ac85afa932e148c9451e177533a8fdf57e9f6fdecc33c1fc`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2500.01, 2495.52, 2496.25, 2504.11, 2504.38, 2511.52, 2511.94, 2515.48, 2516.50, 2518.14 ms.
- Average: 2508.204 ms.
- Decision: reference. The current direct script is back near the restored-default band, so the earlier 2656.684 ms run was not caused by an active Torch/LibTorch route or a promoted candidate stack. It still trails the stable harness baseline by 15.071 ms.

### `direct_train_sm120_live_control_x10_20260521`

- Command: direct `./train-sm120.sh`.
- Binary hash: `train_gpt2cu` `88b5510f494e129bfee579fcb6d4915bf914b4f9a6c09e37ea6cc6a034d1adf6`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2607.87, 2606.00, 2621.11, 2643.30, 2617.62, 2625.12, 2641.86, 2647.97, 2629.67, 2634.24 ms.
- Average: 2629.656 ms.
- Pre-run GPU query: RTX 5090 visible, idle P3, 77.42 W / 575 W, SM/memory clocks 1155/7001 MHz, temperature 33 C, throttle flags `0x0`.
- Post-run GPU query: idle P3, 82.50 W / 575 W, SM/memory clocks 1275/7001 MHz, temperature 40 C, throttle flags `0x0`.
- Cleanup: removed the diagnostic `model_00000010.bin` and `state_00000010_00000.bin` files created by the direct script.
- Decision: regression reference. The slow run is on the restored default binary and unchanged trainer backend mix, so it should not be attributed to an active Torch route, LibTorch route, or leftover candidate compile flag.

### `libtorch_grad_zero_live_retest_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_live_retest_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_libtorch_grad_zero_live_retest_x3_20260521`
- Build flags: `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0`; compile lines included `LLMK_SM120_USE_LIBTORCH_MEMORY` and `LLMK_SM120_USE_LIBTORCH_GRAD_ZERO`.
- Rationale: retest the only remaining Torch operator win with a real trainer call-site under the same degraded runtime state as the live default control.
- Correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2604.84, 2597.65, 2602.69 ms.
- Average: 2600.170 ms.
- Cleanup: candidate checkpoint files were removed by the harness; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject for promotion. This is a same-session current-state improvement over the restored default x3, but it still trails the stable `2493.133 ms` x10 baseline by 107.037 ms and the prior LibTorch grad-zero x10 stability round already regressed to 2495.623 ms.

### `default_after_libtorch_grad_zero_live_retest_x3_20260521`

- Command: direct `./train_gpt2cu` after rebuilding default/no-candidate binaries.
- Trainer output: `log124M/5090_S_default_after_libtorch_grad_zero_live_retest_x3_20260521`
- Binary hash: `train_gpt2cu` `17aa46a59f30a38bdad745f5cba508e71036389af73b847141e29a2e736155bf`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2623.37, 2617.04, 2614.61 ms.
- Average: 2615.825 ms.
- Cleanup: removed the diagnostic `model_00000003.bin` and `state_00000003_00000.bin` files created by the direct control.
- Decision: reference only. The default binary was restored; current-state timing remains degraded and is not a new promotion baseline.

### `combo_libtorch_grad_zero_precompute_scale_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_precompute_scale_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_precompute_scale_x3_20260521`
- Build flags: `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0`, `EXTRA_NVCC_FLAGS=-DLLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW`; compile lines included `LLMK_SM120_USE_LIBTORCH_MEMORY`, `LLMK_SM120_USE_LIBTORCH_GRAD_ZERO`, and `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW`.
- Rationale: combine the current-state LibTorch grad-zero win with the precomputed device AdamW scalar path, which had been better than the per-thread device-sqrt variant, to test whether the two independent step-tail changes compose.
- Correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = precomputed device AdamW scalar`, and `gelu_fusion = 1`.
- Step timings: 2513.09, 2509.34, 2524.45 ms.
- Average: 2516.894 ms.
- Cleanup: candidate checkpoint files were removed by the harness; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. It is much faster than the degraded direct-script band, but the same-session restored default x3 averaged 2502.819 ms, so the combo is slower by 14.075 ms and remains 23.761 ms slower than the stable x10 baseline.

### `default_after_combo_libtorch_grad_zero_precompute_scale_x3_20260521`

- Command: direct `./train_gpt2cu` after rebuilding default/no-candidate binaries.
- Trainer output: `log124M/5090_S_default_after_combo_libtorch_grad_zero_precompute_scale_x3_20260521`
- Binary hash: `train_gpt2cu` `e09409f7cae7a45450a34fb3c3655276e79d663cd474d3a3a27fd53b6b336ffb`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2499.22, 2500.28, 2505.36 ms.
- Average: 2502.819 ms.
- Cleanup: removed the diagnostic `model_00000003.bin` and `state_00000003_00000.bin` files created by the direct control.
- Decision: reference only. Default was restored, and this same-session control rejects the combined Torch/precomputed-scalar candidate as a speed promotion.

### `combo_libtorch_grad_zero_layernorm_bwd1_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_layernorm_bwd1_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_layernorm_bwd1_x3_20260521`
- Build flags: `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0`, `EXTRA_NVCC_FLAGS=-DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`; compile lines included `LLMK_SM120_USE_LIBTORCH_MEMORY`, `LLMK_SM120_USE_LIBTORCH_GRAD_ZERO`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Rationale: combine the strongest prior trainer-callable Torch memory x3 signal with the independent LayerNorm one-block-per-SM near-miss, instead of treating either row as a standalone promotion.
- Correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2658.83, 2559.45, 2539.88 ms.
- Average: 2549.664 ms.
- Cleanup: candidate checkpoint files were removed by the harness; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject for promotion. The combo beat the degraded same-session default control by 48.017 ms, but it is 56.531 ms slower than the stable `2493.133 ms` x10 baseline and the underlying LibTorch grad-zero route already failed x10 stability.

### `default_after_combo_libtorch_grad_zero_layernorm_bwd1_x3_20260521`

- Command: direct `./train_gpt2cu` after rebuilding default/no-candidate binaries.
- Trainer output: `log124M/5090_S_default_after_combo_libtorch_grad_zero_layernorm_bwd1_x3_20260521`
- Binary hash: `train_gpt2cu` `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2575.38, 2614.56, 2580.80 ms.
- Average: 2597.681 ms.
- Cleanup: removed the diagnostic `model_00000003.bin` and `state_00000003_00000.bin` files created by the direct control.
- Decision: reference only. The default binary was restored and the same-session control is degraded, so the combo's current-state win is not evidence for a stable promotion.

### `combo_libtorch_dresidual_zero_bias_wide1024_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_bias_wide1024_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_bias_wide1024_x3_20260521`
- Build flags: `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `SM120_USE_LIBTORCH_GRAD_ZERO=0`, `EXTRA_NVCC_FLAGS=-DLLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024`; compile lines included `LLMK_SM120_USE_LIBTORCH_MEMORY`, `LLMK_SM120_USE_LIBTORCH_DRESIDUAL_ZERO`, and `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024`.
- Rationale: combine the LibTorch dresidual-zero x3 near-miss with the independent wide-bias full-round x3 signal while avoiding the already-rejected LibTorch grad-zero plus wide-bias path.
- Correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2609.07, 2499.87, 2519.59 ms.
- Average: 2509.732 ms.
- Cleanup: candidate checkpoint files were removed by the harness; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject for promotion. The candidate beat the restored same-session default by only 1.349 ms, which is noise-level, and it remains 16.599 ms slower than the stable `2493.133 ms` x10 baseline.

### `default_after_combo_libtorch_dresidual_zero_bias_wide1024_x3_20260521`

- Command: direct `./train_gpt2cu` after rebuilding default/no-candidate binaries.
- Trainer output: `log124M/5090_S_default_after_combo_libtorch_dresidual_zero_bias_wide1024_x3_20260521`
- Binary hash: `train_gpt2cu` `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2506.22, 2507.96, 2514.20 ms.
- Average: 2511.081 ms.
- Cleanup: removed the diagnostic `model_00000003.bin` and `state_00000003_00000.bin` files created by the direct control.
- Decision: reference only. The default binary was restored, and the candidate's small same-session edge is not evidence for a stable training-speed promotion.

### `combo_bias_wide1024_layernorm_bwd1_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_bias_wide1024_layernorm_bwd1_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_bias_wide1024_layernorm_bwd1_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS=-DLLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`; compile lines included both `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024` and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Rationale: combine two native, trainer-callable x3 near-misses that touch independent runtime and LayerNorm kernels, without adding the unstable LibTorch memory route.
- Correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2501.05, 2498.98, 2500.92 ms.
- Average: 2499.946 ms.
- Cleanup: candidate checkpoint files were removed by the harness; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject for promotion. The candidate beat the restored same-session default by 3.385 ms, but it remains 6.813 ms slower than the stable `2493.133 ms` x10 baseline and both components have already failed their own x10 gates.

### `default_after_combo_bias_wide1024_layernorm_bwd1_x3_20260521`

- Command: direct `./train_gpt2cu` after rebuilding default/no-candidate binaries.
- Trainer output: `log124M/5090_S_default_after_combo_bias_wide1024_layernorm_bwd1_x3_20260521`
- Binary hash: `train_gpt2cu` `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2505.40, 2501.56, 2505.10 ms.
- Average: 2503.331 ms.
- Cleanup: removed the diagnostic `model_00000003.bin` and `state_00000003_00000.bin` files created by the direct control.
- Decision: reference only. The default binary was restored, and the native candidate's same-session edge is too small and still behind the stable baseline.

### `direct_train_sm120_with_dmon_x10_20260521`

- Command: `./train-sm120.sh` with concurrent `nvidia-smi dmon -s pucvmt -d 1 -c 40`.
- Captured stdout: `scratch/train-sm120-direct-with-dmon-x10-20260521.log`
- Telemetry log: `scratch/dmon-train-sm120-direct-20260521.log`
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2518.55, 2511.61, 2514.20, 2516.72, 2529.55, 2588.06, 2527.98, 2524.50, 2529.01, 2527.48 ms.
- Average: 2529.901 ms.
- Telemetry: during the steady training window dmon reported SM utilization at 99-100%, power around 574-580 W, GPU temperature rising from 62 C to 69 C, thermal violation `0`, power violation `0`, memory clock `13801 MHz`, and graphics clock mostly 2647-2685 MHz.
- Decision: regression reference. The direct script slowdown aligns with a lower current boost/clock operating point under full load, not with a selected-kernel change.

### `current_default_refresh_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_current_default_refresh_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_current_default_refresh_x10_20260521`
- Build flags: default SM120 `LLMK_SM120_USE_CUBLASLT_GEMM`; no candidate `EXTRA_NVCC_FLAGS`; no LibTorch trainer memory route.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2611.84, 2614.82, 2610.73, 2620.19, 2642.77, 2626.43, 2629.08, 2627.22, 2641.57, 2638.76 ms.
- Average: 2627.950 ms.
- Post-run GPU query: idle `P3`, power 80.86 W of 575 W, SM/memory clocks 1117/7001 MHz, temperature 39 C, active throttle flags `0x0`.
- Decision: regression reference. This is the current default backend mix, but the speed band is far slower than the stable x10 baseline, so it is not a new promotion baseline.

### `restored_default_hash_x3_20260521`

- Command: direct `./train_gpt2cu` after the exact default/no-candidate restore build.
- Trainer output: `log124M/5090_S_codex_sm120_restored_default_hash_x3_20260521`
- Binary hash: `train_gpt2cu` `dba87b1716263f91ac85afa932e148c9451e177533a8fdf57e9f6fdecc33c1fc`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2692.03, 2687.20, 2701.98 ms.
- Average: 2694.590 ms.
- Decision: regression reference. Restoring the known default binary did not recover the stable `2493.133 ms` x10 band, so the active slowdown is not explained by a candidate flag or the broader harness build target set.

### `default_dmon_probe_x3_20260521`

- Command: direct `./train_gpt2cu` with concurrent `nvidia-smi dmon -s pucvmt -d 1 -c 25`.
- Trainer output: `log124M/5090_S_codex_sm120_default_dmon_probe_x3_20260521`
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2628.92, 2619.85, 2628.94 ms.
- Average: 2624.396 ms.
- Telemetry: during the steady training window dmon reported SM utilization at 99-100%, power around 574-577 W, GPU temperature rising from 58 C to 63 C, thermal violation `0`, power violation `0`, memory clock `13801 MHz`, and graphics clock mostly 2707-2775 MHz.
- Decision: regression reference. The slow current band happens while the GPU is fully loaded and not reporting power or thermal violations, so more kernel-combination trials need to be judged against the stable x10 baseline, not only against the degraded current runtime band.

### `current_control_fresh_rebuild_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_current_control_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_current_control_x3_20260521`
- Fresh default build flags: default SM120 `LLMK_SM120_USE_CUBLASLT_GEMM`, no candidate `EXTRA_NVCC_FLAGS`, and no LibTorch trainer flags.
- Built `train_gpt2cu` sha256 `1df684d4313a1edcd812bcf8a74294bf9af4f20d1cea7e49efd368ecaa12d9f9`, size `3060032`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2870.54, 2595.45, 2572.14 ms.
- Average: 2583.795 ms.
- Later correction: after rebuilding `bench_sm120_matmul` without rejected candidate flags, `scratch/bench-sm120-matmul-default-recheck-20260521.log` measured `fcproj dInp+dGeLU` cuBLASLt fused at 1859.02 us, close to the stable artifact's 1841.30 us. The earlier 5009.99 us spot-check was contaminated/noisy and should not be used as a promotion signal.
- Decision: noisy reference only. This run proved a slow state existed, but it did not persist after resetting the benchmark/trainer artifacts.

### `default_post_bench_reset_x3_20260521`

- Command: direct `./train_gpt2cu` with default SM120 build, unique output dir.
- Captured stdout: `scratch/train-sm120-default-post-bench-reset-x3-20260521.log`
- Trainer output: `log124M/5090_S_codex_sm120_default_post_bench_reset_x3_20260521`
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2501.37, 2497.70, 2498.56 ms.
- Average: 2498.128 ms.
- Focused benchmark recheck: `scratch/bench-sm120-matmul-default-recheck-20260521.log` measured `fcproj dInp+dGeLU` cuBLASLt fused at 1859.02 us and TK at 1794.35 us; the cuBLASLt row is close to the stable x10 artifact.
- Decision: reference. Current default performance is near, but still slower than, the stable x10 baseline.

### `direct_train_sm120_after_wide_reset_x10_20260521`

- Command: direct `./train-sm120.sh`, captured at `scratch/train-sm120-default-after-wide-reset-x10-20260521.log`.
- Build reset: rebuilt `train_gpt2cu`, `bench_sm120_runtime`, `test_bias`, and `bench_sm120_matmul` without candidate `EXTRA_NVCC_FLAGS`.
- Binary hash: `train_gpt2cu` `e7abfa54fb5017c53d5351d9d1d6aaa06b66fe4bbefe88d4891ace5063bde618`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2499.84, 2495.44, 2498.85, 2502.50, 2503.08, 2516.34, 2531.87, 2513.74, 2515.51, 2517.92 ms.
- Average: 2510.584 ms.
- Decision: reference. The severe `2656.684 ms` direct-script slowdown recovered after reset, but this path is still slower than the stable x10 harness baseline.

### `combo_libtorch_grad_zero_bias_wide1024_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_bias_wide1024_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_bias_wide1024_x3_20260521`
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2815.97 ms, 2674.42 ms, 2701.92 ms.
- Validator summary: `train_steps=3`, `avg_ms=2688.174`, validation OK.
- Decision: reject. The Torch/LibTorch memory edge seen in focused runtime rows does not survive trainer integration in this composition.

### `combo_cublas_dinp_fc_bias_wide1024_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cublas_dinp_fc_bias_wide1024_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cublas_dinp_fc_bias_wide1024_x3_20260521`
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2734.97 ms, 2631.93 ms, 2569.92 ms.
- Validator summary: `train_steps=3`, `avg_ms=2600.928`, validation OK.
- Decision: reject. The two near-miss native/trainer-callable settings compose poorly in the trainer.

### `combo_layernorm_bwd1_classifier_exp2_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_layernorm_bwd1_classifier_exp2_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_layernorm_bwd1_classifier_exp2_x3_20260521`
- Build flags: `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_SM120_CLASSIFIER_EXP2`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Focused benchmark signal: LayerNorm backward improved to 276.144 us for `C=768`, but classifier loss and fused classifier did not improve materially.
- Step timings: 2510.32 ms, 2501.96 ms, 2507.87 ms.
- Validator summary: `train_steps=3`, `avg_ms=2504.913`, validation OK.
- Decision: reject. Localized CUDA improvements did not survive full trainer integration.

### `combo_cublas_dinp_attproj_layernorm_bwd1_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cublas_dinp_attproj_layernorm_bwd1_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cublas_dinp_attproj_layernorm_bwd1_x3_20260521`
- Build flags: `LLMK_SM120_USE_CUBLAS_DINP_ATTPROJ`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Focused benchmark signal: attention-projection dInput was faster with direct cuBLAS than cuBLASLt for the benchmarked row (`365.49 us` vs `394.27 us`), and LayerNorm backward measured `278.309 us`; the pair still did not improve trainer timing.
- Step timings: 2512.11 ms, 2505.53 ms, 2509.42 ms.
- Validator summary: `train_steps=3`, `avg_ms=2507.472`, validation OK.
- Decision: reject. Focused row wins did not compose into a faster training stack.

### `combo_tk_dgelu_layernorm_bwd1_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521`
- Build flags: `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Focused benchmark signal: `dInp+dGeLU` favored TK over cuBLASLt fused for the GPT-2 `fcproj` row (`1809.820 us` vs `1860.840 us`), and LayerNorm backward measured `272.067 us`; the pair still regressed the trainer.
- Step timings: 2547.77 ms, 2521.33 ms, 2534.30 ms.
- Validator summary: `train_steps=3`, `avg_ms=2527.814`, validation OK.
- Decision: reject. This reinforces that the TK exact-dGELU row is not promotable just because the focused row wins; it needs a trainer-wide scheduling or synchronization fix before another stability gate.

### `tk_dgelu_current_retest_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_tk_dgelu_current_retest_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_tk_dgelu_current_retest_x3_20260521`
- Build flags: `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2757.01, 2720.50, 2623.54 ms.
- Average: 2672.022 ms.
- Decision: reject. The isolated TK fused dGELU trainer route still does not improve end-to-end training, even though the current focused cuBLASLt fused row looked regressed.

### `ge0_current_x3_20260521`

- Command: direct `./train_gpt2cu ... -ge 0 -x 3`.
- Captured stdout: `scratch/train-sm120-ge0-current-x3-20260521.log`
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 0`.
- Step timings: 2609.40, 2639.17, 2605.23 ms.
- Average: 2622.199 ms.
- Decision: reject. The fused GELU route remains required for competitive trainer speed.

### `bias_vec4_retest_x3_20260521`

- Candidate source: temporary opt-in `LLMK_SM120_BIAS_ADD_VEC4` path in `llmc/matmul.cuh`; removed after rejection so the default binary returned to the pre-candidate hash.
- Build flags: `EXTRA_NVCC_FLAGS=-DLLMK_SM120_BIAS_ADD_VEC4`.
- Correctness: `scratch/test-bias-vec4-20260521.log` passed all bias-add aligned/fallback and bias-gradient checks.
- Focused benchmark: `scratch/bench-sm120-runtime-vec4-20260521.log` measured bias-add at 85.413 us for `BT=65536 OC=768` and 560.556 us for `BT=65536 OC=3072`, slower than stable default rows 79.560 us and 536.604 us.
- Trainer stdout: `scratch/train-sm120-bias-vec4-retest-x3-20260521.log`
- Step timings: 2510.66, 2508.18, 2508.85 ms.
- Average: 2508.513 ms.
- Decision: reject. The old vec4 x3 win did not reproduce in the current controlled opt-in path.

### `combo_libtorch_grad_dresidual_zero_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_dresidual_zero_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_grad_dresidual_zero_x3_20260521`
- Build flags: `LLMK_SM120_USE_LIBTORCH_MEMORY`, `LLMK_SM120_USE_LIBTORCH_GRAD_ZERO`, `LLMK_SM120_USE_LIBTORCH_DRESIDUAL_ZERO`.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = Torch C++`, and `gelu_fusion = 1`.
- Focused benchmark signal: runtime rows had shown tiny Torch C++ memset edges, but the linked trainer composition paid large call-site overhead.
- Step timings: 3020.16 ms, 2613.67 ms, 2735.67 ms.
- Validator summary: `train_steps=3`, `avg_ms=2674.668`, validation OK.
- Decision: reject. Do not promote LibTorch memory routes together; they are operator evidence only until a lower-overhead trainer integration exists.

### `bias_wide1024_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_bias_wide1024_x10_20260521`
- Build flags: `EXTRA_NVCC_FLAGS=-DLLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2500.65, 2498.20, 2500.02, 2504.35, 2508.12, 2512.18, 2513.50, 2515.15, 2517.50, 2519.30 ms.
- Average: 2509.814 ms.
- Decision: reject. The earlier 3-step wide-block signal does not survive x10 stability, and it is slower than the current 2493.133 ms stable baseline.

### `cuda_kernel_zero_both_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_cuda_kernel_zero_both_x3_20260521_escalated`
- Trainer output: `log124M/5090_S_codex_sm120_cuda_kernel_zero_both_x3_20260521_escalated`
- Build flags: `EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_USE_CUDA_KERNEL_DRESIDUAL_ZERO"`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = CUDA kernel`, and `gelu_fusion = 1`.
- Step timings: 2522.65, 2499.15, 2500.30 ms.
- Average: 2499.725 ms.
- Decision: reject. The native CUDA-kernel zero route is trainer-callable and avoids Torch integration overhead, but it still does not improve end-to-end training versus the current stable default.

### `default_after_cuda_kernel_zero_hook_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_default_after_cuda_kernel_zero_hook_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_default_after_cuda_kernel_zero_hook_x3_20260521`
- Build flags: default SM120 build, no candidate `EXTRA_NVCC_FLAGS`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2503.46, 2498.39, 2500.91 ms.
- Average: 2499.649 ms.
- Decision: reference. The opt-in CUDA-kernel-zero hooks do not change the unflagged default runtime zeroing path.

### `cuda_kernel_zero_both_block1024_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_cuda_kernel_zero_both_block1024_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_cuda_kernel_zero_both_block1024_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_USE_CUDA_KERNEL_DRESIDUAL_ZERO -DLLMK_SM120_MEMORY_BLOCK_SIZE=1024"`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = CUDA kernel`, and `gelu_fusion = 1`.
- Step timings: 2500.93, 2497.97, 2499.62 ms.
- Average: 2498.795 ms.
- Decision: reject. The 1024-thread memory-kernel variant is a small same-session x3 edge over the unflagged hook build, but the edge is noise-level and still slower than the 2493.133 ms stable x10 baseline.

### `cublaslt_plan_cache_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_cublaslt_plan_cache_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_cublaslt_plan_cache_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS=-DLLMK_SM120_CACHE_CUBLASLT_PLANS`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2506.87, 2503.88, 2506.18 ms.
- Average: 2505.027 ms.
- Decision: reject. The trainer-wide cuBLASLt plan-cache route is slower than the unflagged default controls and does not justify an x10 stability run.

### `disable_cuda_profiler_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_disable_cuda_profiler_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_disable_cuda_profiler_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS=-DLLMK_DISABLE_CUDA_PROFILER`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2502.21, 2497.75, 2498.64 ms.
- Average: 2498.197 ms.
- Decision: reject. Disabling the CUDA profiler calls is a small same-session x3 edge over the unflagged hook-control build, but it is still slower than the current 2493.133 ms stable x10 baseline and does not justify promotion.

### `cublas_dinp_qkv_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_cublas_dinp_qkv_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_cublas_dinp_qkv_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS=-DLLMK_SM120_USE_CUBLAS_DINP_QKV`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2513.69, 2511.68, 2515.61 ms.
- Average: 2513.647 ms.
- Decision: reject. The qkv dInput cuBLAS selector was the remaining trainer-callable microbench flip without a direct smoke, but it is slower than both the same-session default reference and the 2493.133 ms stable x10 baseline.

### `default_after_qkv_selector_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_default_after_qkv_selector_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_default_after_qkv_selector_x3_20260521`
- Build flags: default SM120 build, no candidate `EXTRA_NVCC_FLAGS`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2632.49, 2520.12, 2504.45 ms.
- Average: 2512.286 ms.
- Decision: reference. The same-session default is currently slow, but it remains slightly faster than the qkv cuBLAS dInput selector; default binaries were restored after the rejected candidate.

### `combo_cublas_fc_tk_dgelu_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cublas_fc_tk_dgelu_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cublas_fc_tk_dgelu_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_FC -DLLMK_SM120_USE_TK_FUSED_DGELU_DINP"`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2523.66, 2565.42, 2584.29 ms.
- Average: 2574.852 ms.
- Decision: reject. These two MLP-backward near-misses do not compose; the combined trainer route is much slower than the current stable baseline and slower than the same-session default controls.

### `disable_cuda_profiler_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_disable_cuda_profiler_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_disable_cuda_profiler_x10_20260521`
- Build flags: `EXTRA_NVCC_FLAGS=-DLLMK_DISABLE_CUDA_PROFILER`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2511.39, 2495.51, 2543.14, 2568.75, 2610.24, 2548.38, 2616.64, 2553.50, 2577.45, 2521.18 ms.
- Average: 2559.420 ms.
- Decision: reject. The x10 gate rules out the profiler start/stop path as a training-speed fix; disabling it does not prevent late-step slowdown and is much slower than the current stable x10 baseline.

### `attention_atomic_dq_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_attention_atomic_dq_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_attention_atomic_dq_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS=-DLLMK_SM120_ATOMIC_DQ`.
- Candidate setup: added an atomic-dQ branch to `dev/cuda/bench_sm120_attention.cu` so the benchmark can compile and measure the same generic attention route that the trainer uses when `LLMK_SM120_ATOMIC_DQ` is enabled.
- Correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed. `test_attention` confirmed `SM120 packed-QKV fast path: no` under the candidate.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2917.95, 2966.90, 2923.30 ms.
- Average: 2945.101 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_attention`, `bench_sm120_matmul`, `bench_sm120_runtime`, and `test_attention`; restored `train_gpt2cu` sha256 `dba87b1716263f91ac85afa932e148c9451e177533a8fdf57e9f6fdecc33c1fc`.
- Default attention benchmark after restore: `LLMK_BENCH_REPEATS=3 ./bench_sm120_attention` measured forward 786.763 us and backward 2741.636 us.
- Decision: reject. This is the trainer-shaped attention family, but disabling packed-QKV attention for atomic dQ is far slower than the current packed-TK route and does not justify an x10 gate.

### `combo_libtorch_dresidual_zero_layernorm_bwd1_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_layernorm_bwd1_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_layernorm_bwd1_x3_20260521`
- Build flags: `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `EXTRA_NVCC_FLAGS=-DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Rationale: combine two independent near-miss components instead of treating their isolated scorecard rows as promotion evidence. `libtorch_dresidual_zero_x10` had averaged 2495.957 ms, and `layernorm_bwd_blocks1_x10` had averaged 2501.148 ms, so this tested whether the pair had a positive interaction in the actual trainer.
- Correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, and `gelu_fusion = 1`.
- Step timings: 2503.34, 2496.68, 2499.48 ms.
- Average: 2498.079 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_attention`, `bench_sm120_matmul`, `bench_sm120_runtime`, `test_attention`, and `test_bias`; restored `train_gpt2cu` sha256 `dba87b1716263f91ac85afa932e148c9451e177533a8fdf57e9f6fdecc33c1fc`.
- Decision: reject. The composition was correct and close, but it trailed the 2493.133 ms stable x10 baseline by 4.946 ms and does not justify an x10 gate.

### `combo_cublas_dinp_fc_layernorm_bwd1_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cublas_dinp_fc_layernorm_bwd1_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cublas_dinp_fc_layernorm_bwd1_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_FC -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1"`.
- Rationale: combine the strongest previous FC-only direct-cuBLAS dInput 3-step signal with the independent LayerNorm backward one-block-per-SM candidate, instead of promoting either isolated benchmark row or smoke result alone. Prior evidence: `codex_sm120_round_cublas_dinp_fc_only_20260521` averaged 2490.977 ms over three steps but regressed to 2504.399 ms in x10, while `layernorm_bwd_blocks1_x10` averaged 2501.148 ms.
- Correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Step timings: 2526.20, 2528.61, 2503.12 ms.
- Average: 2515.866 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_attention`, `bench_sm120_matmul`, `bench_sm120_runtime`, `test_attention`, and `test_bias`; restored `train_gpt2cu` sha256 `dba87b1716263f91ac85afa932e148c9451e177533a8fdf57e9f6fdecc33c1fc`.
- Decision: reject. The combined trainer path is much slower than the 2493.133 ms stable x10 baseline and does not justify an x10 gate.

### `nonblocking_main_stream_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_nonblocking_main_stream_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_nonblocking_main_stream_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS=-DLLMK_SM120_NONBLOCKING_MAIN_STREAM`.
- Rationale: test a synchronization-path hypothesis adjacent to the previously promoted backward stream-sync optimization, without changing the default stream behavior unless the trainer smoke proved it.
- Implementation: added an opt-in `cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking)` branch guarded by `KITTENS_SM120 && LLMK_SM120_NONBLOCKING_MAIN_STREAM`; default builds still call `cudaStreamCreate(&main_stream)`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Trainer correctness failed despite focused smokes: the first validation loss was 0.002881 instead of the expected 11.033154 band, final validation loss was 0.000000, and the printed grad norms collapsed to 0.3662, 0.3658, and 0.3483.
- Step timings: 2616.40, 2611.11, 2626.99 ms.
- Average: 2619.052 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_attention`, `bench_sm120_matmul`, `bench_sm120_runtime`, `test_attention`, and `test_bias`.
- Decision: reject on trainer correctness and speed. The non-blocking stream breaks required trainer ordering semantics and is slower than the stable x10 baseline.

### `async_grad_norm_copy_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_async_grad_norm_copy_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_async_grad_norm_copy_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS=-DLLMK_SM120_ASYNC_GRAD_NORM_COPY`.
- Rationale: test whether the per-step grad-norm scalar readback can avoid a broader blocking `cudaMemcpy` synchronization by using `cudaMemcpyAsync(..., main_stream)` followed by `cudaStreamSynchronize(main_stream)`.
- Implementation: added an opt-in SM120 branch in `gpt2_calculate_grad_norm`; default builds still use the original blocking `cudaMemcpy`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2629.80, 2623.50, 2635.29 ms.
- Average: 2629.394889 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_attention`, `bench_sm120_matmul`, `bench_sm120_runtime`, `test_attention`, and `test_bias`; restored `train_gpt2cu` sha256 `48034683e413be75bb5f204b75150166561995a6db695a4e874ba43876df5f30`.
- Decision: reject. The stream-scoped async readback is semantically safe, but slower than the current stable x10 baseline by 136.262 ms and slower than the degraded current band.

### `device_grad_scale_adamw_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_device_grad_scale_adamw_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_device_grad_scale_adamw_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS=-DLLMK_SM120_DEVICE_GRAD_SCALE_ADAMW`.
- Rationale: avoid the host grad-norm scalar copy before AdamW in the timed path by letting the AdamW kernel derive `grad_scale` from the device-resident grad-norm-squared scalar.
- Implementation: split grad-norm reduction into a device-scalar producer plus host readback helper, added an opt-in AdamW kernel wrapper that reads the device scalar, and used it only when skip-update z-score guards are disabled. Default builds still use the original host scalar `grad_scale`.
- Startup confirmed `grad_scale_backend = device AdamW scalar`, `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609930`, and training losses/norms matched the normal smoke band.
- Step timings: 2624.77, 2622.85, 2631.72 ms.
- Average: 2627.288222 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_attention`, `bench_sm120_matmul`, `bench_sm120_runtime`, `test_attention`, and `test_bias`; restored `train_gpt2cu` sha256 `56d3d84e7ba8f4afb855a123ffad60e7c7862239525ade22eeef3f802985807b`.
- Decision: reject. Moving grad-scale computation into AdamW is semantically safe for the current training script path, but it is still much slower than the stable x10 baseline and does not justify an x10 gate.

### `precompute_grad_scale_adamw_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_precompute_grad_scale_adamw_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_precompute_grad_scale_adamw_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS=-DLLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW`.
- Rationale: refine the rejected device-grad-scale AdamW path by moving the `sqrt/div` work out of every AdamW thread into a one-thread precompute kernel, then have AdamW load the precomputed scalar.
- Implementation: added `adamw_compute_grad_scale` plus an opt-in AdamW wrapper that reads `grad_scale_device`; reused `model.accumulated_mean_loss` as the post-backward one-float device scratch after the mean loss was already copied to host.
- Startup confirmed `grad_scale_backend = precomputed device AdamW scalar`, `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609930`, and training losses/norms matched the normal smoke band.
- Step timings: 2627.89, 2615.40, 2614.51 ms.
- Average: 2614.955902 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_attention`, `bench_sm120_matmul`, `bench_sm120_runtime`, `test_attention`, and `test_bias`; restored `train_gpt2cu` sha256 `99ecef5829e6e53461716741dbd2646789125b9b61a88091604bd66744faa920`.
- Decision: reject. Precomputing the device scalar improves over the per-thread device-sqrt variant, but it remains far slower than the stable x10 baseline and does not justify an x10 gate.

### `combo_cuda_zero_block1024_layernorm_bwd1_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cuda_zero_block1024_layernorm_bwd1_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cuda_zero_block1024_layernorm_bwd1_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_USE_CUDA_KERNEL_DRESIDUAL_ZERO -DLLMK_SM120_MEMORY_BLOCK_SIZE=1024 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1"`.
- Rationale: combine two low-overhead native near-misses instead of reusing rejected LibTorch memory routes: CUDA-kernel grad/dresidual zeroing with the best tested memory block size, plus the one-block-per-SM LayerNorm backward setting.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = CUDA kernel`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2635.26, 2616.22, 2618.72 ms.
- Average: 2617.466211 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_attention`, `bench_sm120_matmul`, `bench_sm120_runtime`, `test_attention`, and `test_bias`; restored `train_gpt2cu` sha256 `99ecef5829e6e53461716741dbd2646789125b9b61a88091604bd66744faa920`.
- Decision: reject. The composition is correct, but the low-overhead native near-misses do not interact positively in the current trainer and remain far slower than the stable x10 baseline.

### `cublas_dinp_fcproj_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_cublas_dinp_fcproj_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS=-DLLMK_SM120_USE_CUBLAS_DINP_FCPROJ`.
- Rationale: the GPT-2 MLP-projection dInput row is benchmark-context-sensitive: some prior source states measured direct cuBLAS close to or slightly faster than cuBLASLt, but the current stable selection keeps cuBLASLt. This added a narrow opt-in selector for only `C=3072, OC=768` so the row could be tested in the trainer without changing the default backend mix.
- Implementation: added an opt-in branch in `matmul_sm120_use_cublas_dinp` for `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ`; default builds still route the row through the existing cuBLASLt path.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2610.47, 2691.45, 2610.96 ms.
- Average: 2651.202798 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_attention`, `bench_sm120_matmul`, `bench_sm120_runtime`, `test_attention`, and `test_bias`; restored `train_gpt2cu` sha256 `88b5510f494e129bfee579fcb6d4915bf914b4f9a6c09e37ea6cc6a034d1adf6`.
- Decision: reject. The row is correctness-clean, but direct cuBLAS for MLP-projection dInput is not a trainer improvement under the current source/runtime state and does not justify an x10 gate.

### `combo_project_fastest_dinp_tk_dgelu_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_project_fastest_dinp_tk_dgelu_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_project_fastest_dinp_tk_dgelu_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_ATTPROJ -DLLMK_SM120_USE_CUBLAS_DINP_FC -DLLMK_SM120_USE_TK_FUSED_DGELU_DINP"`.
- Rationale: directly answer whether the fastest trainer-callable matmul microbench rows compose: direct cuBLAS dInput for attention projection and MLP-up, plus TK fused dGELU dInput for MLP projection, while leaving the default CUDA runtime zeroing, host grad-scale, classifier, LayerNorm, and attention routes unchanged.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609922`, and training losses/norms matched the normal smoke band.
- Step timings: 2496.60, 2495.20, 2502.51 ms.
- Average: 2498.856187 ms.
- Same-session default control: after rebuilding default/no-candidate binaries, `scratch/train-sm120-default-after-combo-project-fastest-dinp-tk-dgelu-rerun-x3-20260521.log` averaged 2506.717086 ms with step timings 2516.36, 2506.87, 2506.57 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_runtime`, `bench_sm120_matmul`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: candidate and default-control checkpoint payloads were removed; output dirs retain `DONE_00000003` and `main.log`.
- Decision: reject for promotion. The composition has a real same-session edge of 7.861 ms over the restored default x3, but it still trails the stable `codex_sm120_round_backward_stream_sync_default_x10_20260521` x10 baseline by 5.723 ms, so it does not justify promotion or an x10 gate without another improvement.

### `combo_all_native_winners_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_all_native_winners_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_all_native_winners_x3_20260521`
- Build flags: `EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_ATTPROJ -DLLMK_SM120_USE_CUBLAS_DINP_FC -DLLMK_SM120_USE_TK_FUSED_DGELU_DINP -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1 -DLLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024"`.
- Rationale: broaden the fastest-row composition by adding the two remaining native trainer-callable near-misses that had same-session edges: one-block-per-SM LayerNorm backward and 1024-thread wide bias-add.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609906`, and training losses/norms matched the normal smoke band.
- Step timings: 2498.13, 2495.47, 2500.88 ms.
- Average: 2498.178363 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_runtime`, `bench_sm120_matmul`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: candidate checkpoint payloads were removed by the harness; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject for promotion. This is the best all-native composition tested in this pass, 0.678 ms faster than the narrower fastest dInput/dGeLU combo and 8.539 ms faster than the same-session restored default x3, but it still trails the stable x10 baseline by 5.045 ms.

### `combo_libtorch_grad_zero_all_native_winners_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_all_native_winners_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_all_native_winners_x3_20260521`
- Build flags: `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0`, `EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_ATTPROJ -DLLMK_SM120_USE_CUBLAS_DINP_FC -DLLMK_SM120_USE_TK_FUSED_DGELU_DINP -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1 -DLLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024"`.
- Rationale: combine the remaining Torch trainer route with prior x3 upside, LibTorch grad-zero, with the best native trainer-callable combination, instead of leaving the Torch row as an isolated scorecard result.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609921`, and training losses/norms matched the normal smoke band.
- Step timings: 2496.54, 2494.03, 2499.98 ms.
- Average: 2497.004151 ms.
- Candidate binary hash: `train_gpt2cu` sha256 `82147a41a44ab94f9c6a34e5ea0517c96eb51ec121f3aae720642a70d1e0f113`.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: harness removed `model_00000003.bin` and `state_00000003_00000.bin`; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject for promotion. This is 1.174 ms faster than the all-native winner composition and 2.004 ms faster than the latest default benchmark-warmup x3, but still 3.871 ms slower than the stable x10 baseline, so it does not justify an x10 promotion gate.

### `combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_x3_20260521`
- Build flags: `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0`, `EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_ATTPROJ -DLLMK_SM120_USE_CUBLAS_DINP_FC -DLLMK_SM120_USE_TK_FUSED_DGELU_DINP"`.
- Rationale: remove LayerNorm bwd1 and wide-bias from the LibTorch grad-zero all-native composition to check whether those two native near-miss knobs were hurting the Torch interaction.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609921`, and training losses/norms matched the normal smoke band.
- Step timings: 2497.06, 2495.60, 2497.59 ms.
- Average: 2496.594667 ms.
- Candidate binary hash: `train_gpt2cu` sha256 `ade6de9b4c1a1c5e8143de426313e9fe256d8eb29734101ed600facf16ed3b09`.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: harness removed `model_00000003.bin` and `state_00000003_00000.bin`; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject for promotion. This narrowed stack is 0.409 ms faster than the LibTorch grad-zero all-native composition and 1.584 ms faster than the all-native composition, but it remains 3.462 ms slower than the stable x10 baseline.

### `combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_maxconn1_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_maxconn1_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_fastest_dinp_tk_dgelu_maxconn1_x3_20260521`
- Build flags: `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_ATTPROJ -DLLMK_SM120_USE_CUBLAS_DINP_FC -DLLMK_SM120_USE_TK_FUSED_DGELU_DINP"`.
- Rationale: test whether the runtime scheduling knob that helped the default live band also helps the best LibTorch grad-zero plus dInput/dGELU composition.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609920`, and training losses/norms matched the normal smoke band.
- Step timings: 2494.24, 2496.42, 2498.42 ms.
- Average: 2497.422457 ms.
- Candidate binary hash: `train_gpt2cu` sha256 `def0bf0462cb7c09620914ae53258b8768146ddbb937acd5f3d0cce9d46a658f`.
- Cleanup: harness removed `model_00000003.bin` and `state_00000003_00000.bin`; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The scheduling knob worsened the best dInput/dGELU composition by 0.828 ms and remains 4.289 ms slower than the stable x10 baseline.

### `combo_libtorch_grad_zero_tk_dgelu_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_tk_dgelu_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_tk_dgelu_x3_20260521`
- Build flags: `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0`, `EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP"`.
- Rationale: isolate the two strongest prior x3-only component signals, LibTorch grad-zero and TK exact dGELU dInput, while leaving out the direct-cuBLAS dInput selectors that already failed x10 stability.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609909`, and training losses/norms matched the normal smoke band.
- Step timings: 2895.95, 2694.01, 2623.94 ms.
- Average: 2658.973217 ms.
- Candidate binary hash: `train_gpt2cu` sha256 `d4f047c48030d443c2de418a79d0f8a3087d3b6a653b33225f8976eabd64f5c9`.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The two individually promising x3 routes do not compose: the combination is 165.840 ms slower than the stable x10 baseline and much slower than the broader dInput/dGELU LibTorch composition.

### `libtorch_grad_zero_maxconn1_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_maxconn1_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_libtorch_grad_zero_maxconn1_x3_20260521`
- Build flags: `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`; no extra dInput/dGELU, LayerNorm, or bias-add candidate flags.
- Rationale: test the narrower composition after maxconn1 worsened the all-native and dInput/dGELU winner stacks.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2494.32, 2492.32, 2493.37 ms.
- Average: 2492.843151 ms.
- Candidate binary hash: `train_gpt2cu` sha256 `7bf7450eebff7c2a1c69a7caec468925682d72ca6586fea282462842b0ad1026`.
- Cleanup: harness removed `model_00000003.bin` and `state_00000003_00000.bin`; output dir retains `DONE_00000003` and `main.log`.
- Decision: short-run promotion candidate only. This beat the stable x10 baseline by 0.290 ms, so it required an x10 gate before any trainer default change.

### `libtorch_grad_zero_maxconn1_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_maxconn1_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_libtorch_grad_zero_maxconn1_x10_20260521`
- Build flags: `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`; no extra dInput/dGELU, LayerNorm, or bias-add candidate flags.
- Rationale: x10 stability gate for the short-run winner.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `9.483727`, and training losses/norms matched the normal x10 smoke band.
- Step timings: 2493.39, 2490.46, 2491.64, 2492.93, 2494.08, 2499.13, 2499.37, 2499.41, 2500.58, 2505.83 ms.
- Average: 2497.047398 ms.
- Candidate binary hash: `train_gpt2cu` sha256 `cfc486925c97c78259ab2b014dd91b0159e8976df4f9a870e947136dfd3a3299`.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: harness removed `model_00000010.bin` and `state_00000010_00000.bin`; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject for promotion. The x10 gate averaged 3.914 ms slower than the stable x10 baseline, so the short-run x3 win is not stable enough to use in `train-sm120.sh`.

### `live_default_control_x3_20260521`

- Command: direct `./train_gpt2cu` with the restored default/no-candidate binary.
- Trainer output: `log124M/5090_S_codex_sm120_live_default_control_x3_20260521`
- Captured stdout: `scratch/train-sm120-live-default-control-x3-20260521.log`
- Binary hash: `train_gpt2cu` `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2520.22, 2523.83, 2530.71 ms.
- Average: 2527.265906 ms.
- Cleanup: removed `model_00000003.bin` and `state_00000003_00000.bin`; output dir retains `DONE_00000003` and `main.log`.
- Decision: regression reference. The current live default band is slower than the stable `2493.133 ms` x10 baseline without changing the trainer backend stack.

### `cuda_maxconn1_x3_20260521`

- Command: direct `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train_gpt2cu` with the restored default/no-candidate binary.
- Trainer output: `log124M/5090_S_codex_sm120_cuda_maxconn1_x3_20260521`
- Captured stdout: `scratch/train-sm120-cuda-maxconn1-x3-20260521.log`
- Runtime setting: `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2503.48, 2495.95, 2495.27 ms.
- Average: 2495.606899 ms.
- Cleanup: removed `model_00000003.bin` and `state_00000003_00000.bin`; output dir retains `DONE_00000003` and `main.log`.
- Decision: recovery reference. It improved by 31.659 ms versus the same-session live default x3, but still trailed the stable x10 baseline by 2.474 ms and needed an x10 gate.

### `cuda_maxconn1_x10_20260521`

- Command: direct `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train_gpt2cu` with the restored default/no-candidate binary.
- Trainer output: `log124M/5090_S_codex_sm120_cuda_maxconn1_x10_20260521`
- Captured stdout: `scratch/train-sm120-cuda-maxconn1-x10-20260521.log`
- Runtime setting: `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- Step timings: 2506.60, 2498.67, 2501.53, 2506.65, 2505.85, 2513.02, 2515.79, 2516.19, 2517.21, 2518.98 ms.
- Average: 2510.431263 ms.
- Cleanup: removed `model_00000010.bin` and `state_00000010_00000.bin`; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject for promotion. The runtime knob improves the degraded live band, but the longer x10 gate is 17.298 ms slower than the stable x10 baseline.

### `cuda_maxconn32_x3_20260521`

- Command: direct `CUDA_DEVICE_MAX_CONNECTIONS=32 ./train_gpt2cu` with the restored default/no-candidate binary.
- Trainer output: `log124M/5090_S_codex_sm120_cuda_maxconn32_x3_20260521`
- Captured stdout: `scratch/train-sm120-cuda-maxconn32-x3-20260521.log`
- Runtime setting: `CUDA_DEVICE_MAX_CONNECTIONS=32`.
- Step timings: 2513.52, 2510.03, 2515.55 ms.
- Average: 2512.792230 ms.
- Cleanup: removed `model_00000003.bin` and `state_00000003_00000.bin`; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. This was slower than the `CUDA_DEVICE_MAX_CONNECTIONS=1` recovery check and still slower than the stable x10 baseline.

### `combo_all_native_winners_maxconn1_x3_20260521`

- Command: direct `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train_gpt2cu` after building the best all-native trainer-callable candidate.
- Trainer output: `log124M/5090_S_codex_sm120_combo_all_native_winners_maxconn1_x3_20260521`
- Captured stdout: `scratch/train-sm120-combo-all-native-winners-maxconn1-x3-20260521.log`
- Build flags: `EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_ATTPROJ -DLLMK_SM120_USE_CUBLAS_DINP_FC -DLLMK_SM120_USE_TK_FUSED_DGELU_DINP -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1 -DLLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024"`.
- Runtime setting: `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609919`, and training losses/norms matched the normal smoke band.
- Step timings: 2500.08, 2505.52, 2507.34 ms.
- Average: 2506.428957 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_runtime`, `bench_sm120_matmul`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: removed `model_00000003.bin` and `state_00000003_00000.bin`; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The candidate is correctness-clean, but the best native trainer-callable composition interacts badly with `CUDA_DEVICE_MAX_CONNECTIONS=1`: it is 10.822 ms slower than the maxconn1 default x3 and 13.296 ms slower than the stable x10 baseline.

### `combo_libtorch_dresidual_zero_maxconn1_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_maxconn1_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_maxconn1_x3_20260521`
- Build flags: `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `SM120_USE_LIBTORCH_GRAD_ZERO=0`.
- Runtime setting: `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- Rationale: test whether the near-miss Torch C++ dresidual-zero trainer route interacts positively with the only runtime scheduling knob that recovered the degraded live default band. Prior evidence: `libtorch_dresidual_zero_x10` averaged 2495.957 ms, and `cuda_maxconn1_x3` averaged 2495.607 ms.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on the RTX 5090 path after rerunning outside the sandboxed CUDA-driver context.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2507.84, 2500.41, 2500.44 ms.
- Average: 2500.425458 ms.
- Candidate binary hash: `train_gpt2cu` sha256 `552c690c61c538d07c011a054085b51b2f3d764027c74297893d7c642d3a3615`.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: harness removed `model_00000003.bin` and `state_00000003_00000.bin`; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The composition is correct and faster than some degraded current-state runs, but it is 7.292 ms slower than the stable x10 baseline and 4.819 ms slower than the maxconn1 default x3, so it does not justify an x10 promotion gate.

### `combo_libtorch_dresidual_zero_fastest_dinp_tk_dgelu_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_fastest_dinp_tk_dgelu_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_fastest_dinp_tk_dgelu_x3_20260521`
- Build flags: `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `SM120_USE_LIBTORCH_GRAD_ZERO=0`, `EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_ATTPROJ -DLLMK_SM120_USE_CUBLAS_DINP_FC -DLLMK_SM120_USE_TK_FUSED_DGELU_DINP"`.
- Rationale: combine the near-miss Torch C++ dresidual-zero route with the fastest trainer-callable dInput/dGeLU stack, while omitting the LayerNorm and bias settings that already failed stability.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609927`, and training losses/norms matched the normal smoke band.
- Step timings: 2523.36, 2517.73, 2503.06 ms.
- Average: 2510.395527 ms.
- Candidate binary hash: `train_gpt2cu` sha256 `c0bc384dd02281dc9a4a6dc4e451b97b1304fed8124d20f9baed10cf7ff7e1ea`.
- Default restore: rebuilt default/no-candidate binaries and restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: harness removed `model_00000003.bin` and `state_00000003_00000.bin`; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The composition is correctness-clean but regresses to 17.263 ms slower than the stable x10 baseline and 11.539 ms slower than the native fastest dInput/dGeLU x3 run.

### `guard_adamw_helpers_default_x3_20260521`

- Command: direct `./train_gpt2cu` after temporarily guarding unused opt-in AdamW helper kernels out of default builds.
- Trainer output: `log124M/5090_S_codex_sm120_guard_adamw_helpers_default_x3_20260521`
- Captured stdout: `scratch/train-sm120-guard-adamw-helpers-default-x3-20260521.log`
- Temporary candidate binary hash: `bcaab48d43706fb6369737f6d471b6ed8cf45cd828d8627b57625a10d4f7cd11`.
- Rationale: test whether unused AdamW device-grad-scale helper definitions were affecting the default trainer translation unit enough to shift codegen or runtime speed.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_adamw` passed before the trainer run.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2540.71, 2630.57, 2604.24 ms.
- Average: 2617.405295 ms.
- Default restore: the temporary source guards were reverted manually, `train_gpt2cu` and `test_adamw` were rebuilt, `test_adamw` passed, and `train_gpt2cu` was restored to sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: removed `model_00000003.bin` and `state_00000003_00000.bin`; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. Guarding unused helper kernels did not improve the default trainer; it regressed by 124.272 ms versus the stable x10 baseline.

### `default_native_benchmark_warmup_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_default_native_benchmark_warmup_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_default_native_benchmark_warmup_x3_20260521`
- Command: `RUN_LABEL=codex_sm120_default_native_benchmark_warmup_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 RUN_CURRENT_SELECTION_AUDIT=0 scripts/run_sm120_optimization_round.sh`.
- Rationale: retest whether a no-candidate default run with correctness and native benchmark phases first could reproduce the older `codex_sm120_round_memory_shape_coverage_20260520` x3 average of 2490.206 ms.
- Build/runtime settings: default SM120 native trainer mix, `SM120_USE_LIBTORCH_GRAD_ZERO=0`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0`, `SM120_USE_LIBTORCH_MEMORY=0`, and no candidate `EXTRA_NVCC_FLAGS`.
- Binary hash: current default-source `train_gpt2cu` sha256 `9edf24a2c21d78a60c1c8f9b3c6b81053e6c0526cac85bd5e17f8aa032e2af0e`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark phase recorded 95 benchmark rows before training.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2504.52, 2497.49, 2500.53 ms.
- Average: 2499.008536 ms.
- Validator summary: `benchmarks=95`, `train_steps=3`, `avg_ms=2499.009`, validation OK.
- Cleanup: harness removed `model_00000003.bin` and `state_00000003_00000.bin`; output dir retains `DONE_00000003` and `main.log`.
- Decision: reference only. Native benchmark/correctness preconditioning did not recover the older 2490.206 ms x3 band and remains 5.876 ms slower than the stable x10 baseline.

### `attention_dprep3_current_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_attention_dprep3_current_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_attention_dprep3_current_x3_20260521`
- Command: `RUN_LABEL=codex_sm120_attention_dprep3_current_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DPREP_WARPS=3" scripts/run_sm120_optimization_round.sh`.
- Rationale: retest the older dprep=3 attention-prep signal under the current cuBLASLt-backed trainer mix instead of assuming the previous pure-TK-era result still composes.
- Build/runtime settings: current default native trainer mix plus `LLMK_SM120_DPREP_WARPS=3`; no Torch/LibTorch trainer memory route and no runtime scheduling override.
- Binary hash: candidate `train_gpt2cu` sha256 `6ec1cd3556a23c442d7fcb2c2dec6504e9809eba2f75b1596523c768c307b52d`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2527.38, 2493.35, 2495.17 ms.
- Average: 2494.256735 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The candidate is correctness-clean but `1.124 ms` slower than the stable `2493.133 ms` x10 baseline, so the older dprep=3 signal does not carry into the current trainer stack and does not justify an x10 gate.

### `cublaslt_ws256_heur16_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_cublaslt_ws256_heur16_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_cublaslt_ws256_heur16_x3_20260521`
- Command: `RUN_LABEL=codex_sm120_cublaslt_ws256_heur16_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0 EXTRA_NVCC_FLAGS="-DLLMK_SM120_CUBLASLT_WORKSPACE_MB=256 -DLLMK_SM120_CUBLASLT_HEURISTIC_RESULTS=16" scripts/run_sm120_optimization_round.sh`.
- Rationale: test whether the trainer-wide cuBLASLt rows benefit from a larger workspace and wider heuristic result set after the epilogue-index probe showed the default index was best only within the current returned algorithms.
- Build/runtime settings: current default native trainer mix, default CUDA runtime zeroing, default host scalar AdamW grad scale, cuBLASLt workspace increased from 128 MB to 256 MB, and heuristic results increased from 8 to 16.
- Binary hash: candidate `train_gpt2cu` sha256 `77fa411177a4d1d693fde3483756777c31a9619c57ab698bd733f50f093eb0f7`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2525.00, 2523.69, 2495.88 ms.
- Average: 2509.786487 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. Widening cuBLASLt search/workspace is correctness-clean but `16.653 ms` slower than the stable x10 baseline and slower than the prior plan-cache rejection, so keep the default cuBLASLt workspace and heuristic breadth.

### `combo_cublas_dinp_attproj_fc_dprep3_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cublas_dinp_attproj_fc_dprep3_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cublas_dinp_attproj_fc_dprep3_x3_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_cublas_dinp_attproj_fc_dprep3_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_ATTPROJ -DLLMK_SM120_USE_CUBLAS_DINP_FC -DLLMK_SM120_DPREP_WARPS=3" scripts/run_sm120_optimization_round.sh`.
- Rationale: test whether the closest x10 direct-cuBLAS dInput selector near-miss composes with the older attention dprep=3 near-miss. The individual candidates were both correctness-clean but not promotable alone.
- Build/runtime settings: current default native/cuBLASLt trainer mix, direct cuBLAS dInput for attention projection and MLP-up rows, `LLMK_SM120_DPREP_WARPS=3`, default CUDA runtime zeroing, default host scalar AdamW grad scale, and no LibTorch trainer memory route.
- Binary hash: candidate `train_gpt2cu` sha256 `9ea79da33b882d1aafe889fed0a30c7ac5e665090bac1877986738f80cdc15d4`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2577.58, 2540.39, 2546.56 ms.
- Average: 2543.476224 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The two near-misses do not compose: the candidate is correctness-clean but `50.343 ms` slower than the stable `2493.133 ms` x10 baseline, so no x10 gate is justified.

### `combo_libtorch_grad_zero_maxconn1_dprep3_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_maxconn1_dprep3_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_maxconn1_dprep3_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_combo_libtorch_grad_zero_maxconn1_dprep3_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=1 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DPREP_WARPS=3" scripts/run_sm120_optimization_round.sh`.
- Rationale: combine the only short-run LibTorch memory route that had beaten the stable baseline with the runtime scheduling knob and the closest attention dprep near-miss, then require x10 proof before any promotion.
- Build/runtime settings: current default native/cuBLASLt trainer mix, Torch C++ grad-zero route, CUDA runtime dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and `LLMK_SM120_DPREP_WARPS=3`.
- Binary hash: candidate `train_gpt2cu` sha256 `a30913635da6c856a3a697d35a461df80dc9940d285efd2b827d9ddc190630d7`.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2490.31, 2485.28, 2488.00 ms.
- Average: 2486.639023 ms.
- Decision: gate with x10. The short run was `6.494 ms` faster than the stable x10 baseline, but prior short-run wins failed longer gates.

### `combo_libtorch_grad_zero_maxconn1_dprep3_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_maxconn1_dprep3_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_maxconn1_dprep3_x10_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_combo_libtorch_grad_zero_maxconn1_dprep3_x10_20260521 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=1 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DPREP_WARPS=3" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: same stack as the x3 gate: Torch C++ grad-zero route, CUDA runtime dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and `LLMK_SM120_DPREP_WARPS=3`.
- Binary hash: candidate `train_gpt2cu` sha256 `8cdf5d20d1bb9f9938ef66455fdfe4faf8f4af4cdf450d4f48a7cff68028b3cc`.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `9.483727`, and training losses/norms matched the normal smoke band.
- Step timings: 2489.15, 2484.62, 2491.76, 2491.92, 2491.94, 2495.55, 2517.49, 2545.70, 2500.21, 2562.27 ms.
- Average: 2509.051694 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject. The x10 gate is `15.919 ms` slower than the stable `2493.133 ms` baseline and shows late-step drift, so do not promote the mixed Torch grad-zero, maxconn1, and dprep=3 stack.

### `combo_libtorch_grad_zero_dprep3_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_dprep3_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_dprep3_x3_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_libtorch_grad_zero_dprep3_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=1 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DPREP_WARPS=3" scripts/run_sm120_optimization_round.sh`.
- Rationale: isolate the maxconn-free version of the previous best short-run composition to test whether `CUDA_DEVICE_MAX_CONNECTIONS=1` caused the x10 late drift while preserving Torch C++ grad-zero and the dprep=3 attention near-miss.
- Build/runtime settings: current default native/cuBLASLt trainer mix, Torch C++ grad-zero route, CUDA runtime dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, and `LLMK_SM120_DPREP_WARPS=3`.
- Binary hash: candidate `train_gpt2cu` sha256 `a85b2062c5a853b33e0ec3abff24136fa5077b38152c88847a4b7962d5bf0e73`.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2489.39, 2483.60, 2487.04 ms.
- Average: 2485.321999 ms.
- Decision: gate with x10. The short run was `7.811 ms` faster than the stable x10 baseline, but promotion still requires a longer stability run.

### `combo_libtorch_grad_zero_dprep3_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_dprep3_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_dprep3_x10_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_libtorch_grad_zero_dprep3_x10_20260521 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=1 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DPREP_WARPS=3" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: same stack as the x3 gate: Torch C++ grad-zero route, CUDA runtime dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, and `LLMK_SM120_DPREP_WARPS=3`.
- Binary hash: candidate `train_gpt2cu` sha256 `7a4f77cdabc91130a5ca8ef58ebba782008678f151f618879ac95ecfc0fdfec9`.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `9.483727`, and training losses/norms matched the normal smoke band.
- Step timings: 2493.59, 2490.94, 2490.73, 2493.10, 2493.85, 2499.34, 2499.97, 2502.37, 2501.70, 2505.41 ms.
- Average: 2497.489823 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject. Removing `CUDA_DEVICE_MAX_CONNECTIONS=1` reduced the x10 late-step drift versus the three-way composition, but the candidate is still `4.357 ms` slower than the stable `2493.133 ms` x10 baseline, so do not promote Torch grad-zero plus dprep=3.

### `combo_libtorch_dresidual_zero_dprep3_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_dprep3_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_dprep3_x3_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_libtorch_dresidual_zero_dprep3_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DPREP_WARPS=3" scripts/run_sm120_optimization_round.sh`.
- Rationale: combine the Torch C++ dresidual-zero near-miss with the dprep=3 attention near-miss after grad-zero+dprep remained too slow at x10.
- Build/runtime settings: current default native/cuBLASLt trainer mix, CUDA runtime grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, and `LLMK_SM120_DPREP_WARPS=3`.
- Binary hash: candidate `train_gpt2cu` sha256 `0ed88ed4274deb9c41d8bacf9bb51b7ec9dbef651dd511ac7fff30f64aba7939`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2494.11, 2488.47, 2488.72 ms.
- Average: 2488.596797 ms.
- Decision: gate with x10. The short run was `4.536 ms` faster than the stable x10 baseline, but prior short-run wins have not survived stability gates.

### `combo_libtorch_dresidual_zero_dprep3_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_dprep3_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_dprep3_x10_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_libtorch_dresidual_zero_dprep3_x10_20260521 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DPREP_WARPS=3" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: same stack as the x3 gate: CUDA runtime grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, and `LLMK_SM120_DPREP_WARPS=3`.
- Binary hash: candidate `train_gpt2cu` sha256 `da88989d606f570f3ac2e6791124bb0c75e86244bca78a8b101ade68fc6d7a81`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `9.483727`, and training losses/norms matched the normal smoke band.
- Step timings: 2492.97, 2485.33, 2487.25, 2490.93, 2493.26, 2495.76, 2497.17, 2499.66, 2502.79, 2500.22 ms.
- Average: 2494.706445 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject near-miss. This is the closest recent x10 composition, but it remains `1.573 ms` slower than the stable `2493.133 ms` x10 baseline, so it is not a significant training improvement and should not replace the default trainer.

### `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x3_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_DPREP_WARPS=3 -DLLMK_SM120_MEMORY_BLOCK_SIZE=1024" scripts/run_sm120_optimization_round.sh`.
- Rationale: take the closest recent x10 stack, Torch C++ dresidual-zero plus attention `dprep=3`, and replace only the gradient clear path with the native CUDA zero kernel at block size 1024.
- Build/runtime settings: current default native/cuBLASLt trainer mix, CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, `LLMK_SM120_DPREP_WARPS=3`, and `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`.
- Binary hash: candidate `train_gpt2cu` sha256 `4cc977ab1a5f271b262aa14a384cd036b68dc953d524e3f90839d1e5f1a7098c`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2491.48, 2484.38, 2487.79 ms.
- Average: 2486.083984 ms.
- Decision: gate with x10. The short run was `7.049 ms` faster than the stable x10 baseline and `2.513 ms` faster than the prior Torch dresidual-zero plus dprep=3 x3 gate.

### `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x10_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x10_20260521 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_DPREP_WARPS=3 -DLLMK_SM120_MEMORY_BLOCK_SIZE=1024" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: same stack as the x3 gate: CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, `LLMK_SM120_DPREP_WARPS=3`, and `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`.
- Binary hash: candidate `train_gpt2cu` sha256 `2252c7d16e72813d52a3354c5537977a074ff22fb2acfec330f905164c187dbf`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `9.483727`, and training losses/norms matched the normal smoke band.
- Step timings: 2490.11, 2484.67, 2487.77, 2490.26, 2491.60, 2494.51, 2494.74, 2496.89, 2506.30, 2502.14 ms.
- Average: 2494.319916 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `66c0932a876c57052cdc07d5411e70c8235c237ae600ff7081b289aaa066f002`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject near-miss. This improves the prior Torch dresidual-zero plus dprep=3 x10 by `0.387 ms`, but it remains `1.187 ms` slower than the stable `2493.133 ms` x10 baseline, so it is not a significant training improvement and should not replace the default trainer.

### `combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x3_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DPREP_WARPS=3 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Rationale: combine the closest Torch C++ dresidual-zero plus dprep=3 stack with the LayerNorm backward one-block-per-SM candidate to test whether independent near-misses compose.
- Build/runtime settings: current default native/cuBLASLt trainer mix, CUDA runtime grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, `LLMK_SM120_DPREP_WARPS=3`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Binary hash: candidate `train_gpt2cu` sha256 `434166f391dd27f3722041592dca4874b562360feb29d52278881a95d209deae`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2489.24, 2482.18, 2488.76 ms.
- Average: 2485.469937 ms.
- Decision: gate with x10. This was the best x3 composition in the current pass, beating the stable x10 baseline by `7.663 ms`.

### `combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x10_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x10_20260521 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DPREP_WARPS=3 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: same stack as the x3 gate: CUDA runtime grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, `LLMK_SM120_DPREP_WARPS=3`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Binary hash: candidate `train_gpt2cu` sha256 `13b87de83a1b6bd8930aab835b21ef7aa294f5a3ef5270add8de11b8dfc1c91a`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `9.483727`, and training losses/norms matched the normal smoke band.
- Step timings: 2492.42, 2486.72, 2489.17, 2491.13, 2494.81, 2494.53, 2496.90, 2498.66, 2499.77, 2503.39 ms.
- Average: 2495.009210 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `17970026dbad81d082f4249c4e927eba315032b3a698fdeb05125e8d73877f89`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject near-miss. The short-run win did not survive the stability gate; this stack is `1.876 ms` slower than the stable `2493.133 ms` x10 baseline, so it is not a significant training improvement and should not replace the default trainer.

### `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x3_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_DPREP_WARPS=3 -DLLMK_SM120_MEMORY_BLOCK_SIZE=1024 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Rationale: combine the two strongest recent x3 signals: CUDA-kernel grad zero plus Torch dresidual-zero plus dprep=3, and Torch dresidual-zero plus dprep=3 plus LayerNorm bwd1.
- Build/runtime settings: current default native/cuBLASLt trainer mix, CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Binary hash: candidate `train_gpt2cu` sha256 `767ce9a93a08717d36a880954381d8ae05e2ea550559669b4971cb22400d23b3`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2490.95, 2483.54, 2485.26 ms.
- Average: 2484.399796 ms.
- Decision: gate with x10. This was the best x3 composition in the current pass, beating the stable x10 baseline by `8.733 ms`.

### `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x10_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x10_20260521 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_DPREP_WARPS=3 -DLLMK_SM120_MEMORY_BLOCK_SIZE=1024 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: same stack as the x3 gate: CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Binary hash: candidate `train_gpt2cu` sha256 `a8cd1f3cd70111769cf3c979550bedb6cfc90593e8b247c6ad7ef0579668a125`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `9.483727`, and training losses/norms matched the normal smoke band.
- Step timings: 2491.70, 2487.25, 2488.64, 2489.39, 2493.78, 2493.11, 2495.83, 2497.74, 2498.62, 2499.26 ms.
- Average: 2493.735234 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `d40f2eceabad9ec89cdf873479cf997a5bd780b34900e569eabcf9d072f96c34`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject near-miss. This is the closest current composed x10 result, but it remains `0.602 ms` slower than the stable `2493.133 ms` x10 baseline, so it is not a significant training improvement and should not replace the default trainer.

### `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x3_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_DPREP_WARPS=4 -DLLMK_SM120_MEMORY_BLOCK_SIZE=1024 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Rationale: retest the four-way near-miss with the source-default dprep warp count to see whether dprep=4 trades a slower x3 for less x10 late drift.
- Build/runtime settings: current default native/cuBLASLt trainer mix, CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, `LLMK_SM120_DPREP_WARPS=4`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Binary hash: candidate `train_gpt2cu` sha256 `72254964f04d1fb58669138275f9509ebd899949c8f4ee0a963ae0abc9ffdbce`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2490.32, 2484.10, 2488.38 ms.
- Average: 2486.240149 ms.
- Decision: gate with x10. The source-default dprep=4 variant was slower than the dprep=3 four-way x3 but still beat the stable x10 baseline by `6.893 ms`.

### `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x10_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep4_layernorm_bwd1_block1024_x10_20260521 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_DPREP_WARPS=4 -DLLMK_SM120_MEMORY_BLOCK_SIZE=1024 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: same stack as the x3 gate: CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, `LLMK_SM120_DPREP_WARPS=4`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Binary hash: candidate `train_gpt2cu` sha256 `8a1f37a6b67ad3aabac298593c1535a4f9022d0a0f7639766d31a9713a3aa74e`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `9.483727`, and training losses/norms matched the normal smoke band.
- Step timings: 2492.29, 2483.27, 2487.61, 2489.64, 2495.68, 2496.26, 2496.36, 2497.21, 2498.98, 2500.70 ms.
- Average: 2493.969838 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `aa904a1a5d3ca24a9a370d410a3b73665e64a9a09e9344daaef633427997ad9e`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject near-miss. Dprep=4 did not reduce late-step drift versus the dprep=3 four-way stack; this run is `0.837 ms` slower than the stable `2493.133 ms` x10 baseline, so it is not a significant training improvement and should not replace the default trainer.

### `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x3_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_DPREP_WARPS=3 -DLLMK_SM120_MEMORY_BLOCK_SIZE=512 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Rationale: retest the closest four-way composed stack with a 512-thread memory zero block to see whether the zero-kernel block-size setting was causing x10 tail drift.
- Build/runtime settings: current default native/cuBLASLt trainer mix, CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=512`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Binary hash: candidate `train_gpt2cu` sha256 `12a73a4996cd17c9230aa1e5941cfd49ca138fc099729315ba2cbae37b793db6`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2490.97, 2483.88, 2486.76 ms.
- Average: 2485.323906 ms.
- Decision: gate with x10. The stack beat the stable x10 baseline by `7.809 ms` over x3, so it needed a longer stability run before any default change.

### `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x10_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x10_20260521 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_DPREP_WARPS=3 -DLLMK_SM120_MEMORY_BLOCK_SIZE=512 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: same stack as the x3 gate: CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=512`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Binary hash: candidate `train_gpt2cu` sha256 `64414fc62eae5148f4845c81e7974db5af1f01b77b68912f3852ee91dff24559`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `9.483727`, and training losses/norms matched the normal smoke band.
- Step timings: 2497.13, 2485.52, 2486.60, 2489.27, 2499.71, 2494.10, 2495.73, 2497.79, 2497.96, 2500.92 ms.
- Average: 2494.176679 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `b350a6626f320b170be8d090d47716c169009b92f7b4b418c92f5a3ef0b99d2c`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject near-miss. Block512 is slower than the block1024 four-way x10 and remains `1.044 ms` slower than the stable `2493.133 ms` x10 baseline, so it is not a significant training improvement and should not replace the default trainer.

### `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x3_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_DPREP_WARPS=3 -DLLMK_SM120_MEMORY_BLOCK_SIZE=256 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Rationale: complete the zero-kernel block-size sweep around the closest four-way stack with the 256-thread block size.
- Build/runtime settings: current default native/cuBLASLt trainer mix, CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=256`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Binary hash: candidate `train_gpt2cu` sha256 `42e4aafa0d36ed6456f74876591eebe6086df5ee79695c9f24f3ea2e504963a0`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2490.40, 2483.16, 2488.66 ms.
- Average: 2485.912204 ms.
- Decision: gate with x10. The stack beat the stable x10 baseline by `7.221 ms` over x3, so it needed a longer stability run before any default change.

### `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x10_20260521`
- Command: `RUN_LABEL=codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x10_20260521 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_DPREP_WARPS=3 -DLLMK_SM120_MEMORY_BLOCK_SIZE=256 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: same stack as the x3 gate: CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, default CUDA scheduling, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=256`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Binary hash: candidate `train_gpt2cu` sha256 `4dd1d7a44f594b937c30badacc64290dc2dca2c3e3059e98cc2fa9926a6c271b`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `9.483727`, and training losses/norms matched the normal smoke band.
- Step timings: 2491.54, 2485.61, 2490.56, 2489.97, 2489.62, 2525.57, 2506.49, 2497.52, 2506.83, 2501.96 ms.
- Average: 2499.344932 ms.
- Default restore: rebuilt default/no-candidate `train_gpt2cu`, `bench_sm120_matmul`, `bench_sm120_runtime`, `bench_sm120_attention`, `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`; restored `train_gpt2cu` sha256 `4380d98cdadeb9dad2bbe85923ed4dc438499dc8d06ec3f68b9ce4fa4a3a0680`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject. Block256 is slower than the block512 and block1024 four-way x10 gates and remains `6.212 ms` slower than the stable `2493.133 ms` x10 baseline, so it is not a significant training improvement and should not replace the default trainer.

### `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_DPREP_WARPS=3 -DLLMK_SM120_MEMORY_BLOCK_SIZE=1024 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Rationale: test whether `CUDA_DEVICE_MAX_CONNECTIONS=1` helps the closest four-way stack after the memory-block sweep showed block1024 was still the best rejected variant.
- Build/runtime settings: current default native/cuBLASLt trainer mix, CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Binary hash: candidate `train_gpt2cu` sha256 `71ef30090893dd5d987b1473dec9701ce82f18bbd5dd878338ab983aa5131ea7`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2485.38, 2480.72, 2481.61 ms.
- Average: 2481.166482 ms.
- Decision: gate with x10. This was the best x3 composed stack in the pass, beating the stable x10 baseline by `11.967 ms`.

### `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x10_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x10_20260521 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO -DLLMK_SM120_DPREP_WARPS=3 -DLLMK_SM120_MEMORY_BLOCK_SIZE=1024 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: same stack as the x3 gate.
- Binary hash: candidate `train_gpt2cu` sha256 `799c99b2afd4ad4dd0b1dcba9f497be2bfbf0733487e6ef9c05599657972aa34`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `9.483727`, and training losses/norms matched the normal smoke band.
- Step timings: 2489.45, 2481.76, 2485.72, 2489.58, 2488.89, 2491.85, 2494.07, 2493.16, 2495.90, 2497.54 ms.
- Average: 2490.940015 ms.
- Decision: promote only after confirmation. This x10 gate beat the stable `2493.133 ms` baseline by `2.193 ms`, but a second x10 was required because earlier maxconn1 candidates had been noisy.

### `combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_confirm_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_confirm_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_confirm_x10_20260521`
- Command: same stack as the first maxconn1 x10 gate, with a fresh run label.
- Binary hash: candidate `train_gpt2cu` sha256 `99d2163520d1a425de0ec26528942a54984191fa4d41fb14672fc6014833a8ef`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `9.483727`, and training losses/norms matched the normal smoke band.
- Step timings: 2489.36, 2480.62, 2484.12, 2487.12, 2488.22, 2491.75, 2493.29, 2495.13, 2495.49, 2498.15 ms.
- Average: 2490.431786 ms.
- Decision: promote. This confirmation beat the stable x10 baseline by `2.701 ms`, so the stack was wired into SM120 defaults and `train-sm120.sh`.

### `direct_train_sm120_promoted_fast_default_x10_20260521`

- Command: direct `./train-sm120.sh` after source promotion and an ordinary SM120 rebuild with no candidate `EXTRA_NVCC_FLAGS`.
- Output directory: `log124M/5090_S`
- Build defaults: `SM120_FAST_TRAINER=1`, `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Runtime setting: `train-sm120.sh` exports `CUDA_DEVICE_MAX_CONNECTIONS=1` unless already set by the caller.
- Binary hash: promoted default `train_gpt2cu` sha256 `84c7f61562552428e47ff6d726f24c9c33b4275dbf283a25cbfc7c9bb18bbeb6`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2485.35, 2480.09, 2484.34, 2486.14, 2489.45, 2489.98, 2490.78, 2492.04, 2493.65, 2495.10 ms.
- Average: 2489.062124 ms.
- Decision: direct promotion verified. The user-facing script now beats the previous stable x10 baseline by `4.071 ms` and improves the earlier direct-script restored-default reference by `19.142 ms`.

### `direct_train_sm120_promoted_slow_dmon_x10_20260521`

- Command: direct `./train-sm120.sh` on the restored promoted-default binary, with concurrent `nvidia-smi dmon`.
- Captured stdout: `scratch/train-sm120-promoted-slow-dmon-20260521.log`
- Telemetry: `scratch/dmon-train-sm120-promoted-slow-20260521.log`
- Binary hash: restored promoted-default `train_gpt2cu` sha256 `45a26966f39b30de7310c0d0dabb6103461358a4ad643e54f853e6f420d51017`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Telemetry during steady training showed `99-100%` SM utilization, `573-575 W`, `55-64 C`, pclk about `2707-2782 MHz`, memory clock `13801 MHz`, and `pviol=0`, `tviol=0`.
- Step timings: 2611.43, 2600.34, 2603.51, 2608.27, 2606.15, 2608.08, 2616.97, 2615.22, 2621.26, 2619.04 ms.
- Average: 2610.981014 ms.
- Decision: regression reference. The selected components did not silently change; this is the promoted direct-script path running in a slower current runtime band.

### `promoted_bias_wide1024_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_bias_wide1024_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_bias_wide1024_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_bias_wide1024_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: promoted SM120 fast default plus the wide-bias block-size override: CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024`.
- Binary hash: candidate `train_gpt2cu` sha256 `9a9ac080467c9796956cf7f909737ce6e11f4ae0144aaf6c7063ae89a21fdf20`; restored promoted-default `train_gpt2cu` sha256 `45a26966f39b30de7310c0d0dabb6103461358a4ad643e54f853e6f420d51017`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2577.40, 2580.67, 2635.40 ms.
- Average: 2608.035564 ms.
- Default restore: rebuilt the promoted SM120 default binaries without `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024`; the restore build line retained `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, and `LLMK_SM120_USE_LIBTORCH_DRESIDUAL_ZERO`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The add-on regressed by `118.973 ms` versus the promoted direct-script x10 average `2489.062 ms` and by `114.903 ms` versus the previous stable x10 baseline `2493.133 ms`, so it does not get an x10 gate and must not replace the promoted trainer.

### `promoted_no_maxconn_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_no_maxconn_x3_20260521`
- Build/runtime settings: promoted SM120 fast default with `CUDA_DEVICE_MAX_CONNECTIONS` explicitly unset.
- Binary hash: candidate `train_gpt2cu` sha256 `2b9712f285a626ecab9ff0485e62a28232ff66609c2c0318b82d3ae3cc050188`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Step timings: 2602.56, 2596.57, 2599.35 ms.
- Average: 2597.961307 ms.
- Decision: gate with x10. This beat the current slow maxconn1 direct telemetry band on x3, but it remained far slower than the promoted direct proof and needed a longer run before changing `train-sm120.sh`.

### `promoted_no_maxconn_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_no_maxconn_x10_20260521`
- Build/runtime settings: same as the no-maxconn x3 gate.
- Binary hash: candidate `train_gpt2cu` sha256 `2e10f52c198c1bbaa43f79dfa3cda487013a403e240deeca5072c5a16513f6cb`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Step timings: 2610.20, 2601.27, 2639.09, 2609.12, 2690.18, 2622.10, 2648.51, 2614.98, 2620.51, 2641.60 ms.
- Average: 2631.929292 ms.
- Decision: reject. The no-maxconn x10 gate is `20.948 ms` slower than the current maxconn1 direct telemetry reference `2610.981 ms`, and `142.867 ms` slower than the promoted direct proof `2489.062 ms`; keep the script default at `CUDA_DEVICE_MAX_CONNECTIONS=1`.

### `promoted_maxconn2_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_maxconn2_x3_20260521`
- Build/runtime settings: promoted SM120 fast default with `CUDA_DEVICE_MAX_CONNECTIONS=2`.
- Binary hash: candidate `train_gpt2cu` sha256 `c9ecff72fb44849fe20bee7581361f04076e7464b02e0b0aadc3b3c04d6adc5d`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Step timings: 2622.58, 2618.88, 2608.54 ms.
- Average: 2613.712430 ms.
- Decision: reject. `CUDA_DEVICE_MAX_CONNECTIONS=2` is slower than no-maxconn x3 and slower than the current maxconn1 direct telemetry reference.

### `promoted_cuda_dresidual_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_cuda_dresidual_x3_20260521`
- Build/runtime settings: promoted compile-time stack with CUDA-kernel grad-zero, dprep=3, memory block1024, LayerNorm bwd1, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and CUDA runtime dresidual-zero instead of the promoted Torch C++ route.
- Binary hash: candidate `train_gpt2cu` sha256 `142552ea94123b06da93d651756367f20d36507b9e7f07e94d8acb2ea53e2b43`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Step timings: 2664.09, 2637.69, 2610.85 ms.
- Average: 2624.265790 ms.
- Decision: reject. Replacing the promoted Torch C++ residual clear with the CUDA runtime route regressed versus current promoted-stack references, so Torch C++ remains the selected dresidual-zero route.

### `promoted_cublas_dinp_attproj_fc_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_attproj_fc_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_cublas_dinp_attproj_fc_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_cublas_dinp_attproj_fc_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_ATTPROJ -DLLMK_SM120_USE_CUBLAS_DINP_FC" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: promoted SM120 fast default plus direct-cuBLAS dInput selectors for attention projection and MLP-up: CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_SM120_USE_CUBLAS_DINP_ATTPROJ`, and `LLMK_SM120_USE_CUBLAS_DINP_FC`.
- Binary hash: candidate `train_gpt2cu` sha256 `335b4c41ae66898f142843075e5151ab2561d72db618cd2ea4beb47dcaccd6a7`; restored promoted-default `train_gpt2cu` sha256 `d061c1e460c9614d68e728f28840887c77ca38d0746596e7d0c1e6777f8b5879`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2601.04, 2595.76, 2642.05 ms.
- Average: 2618.904829 ms.
- Default restore: rebuilt the promoted SM120 default binaries without `LLMK_SM120_USE_CUBLAS_DINP_ATTPROJ` or `LLMK_SM120_USE_CUBLAS_DINP_FC`; the restore build line retained `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, and `LLMK_SM120_USE_LIBTORCH_DRESIDUAL_ZERO`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The candidate is `7.924 ms` slower than the current slow maxconn1 telemetry reference `2610.981 ms`, `129.843 ms` slower than the promoted direct proof `2489.062 ms`, and `125.772 ms` slower than the prior stable x10 baseline `2493.133 ms`.

### `promoted_classifier_exp2_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_classifier_exp2_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_classifier_exp2_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_classifier_exp2_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_CLASSIFIER_EXP2" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: promoted SM120 fast default plus the classifier `exp2` path: CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `LLMK_SM120_CLASSIFIER_EXP2`.
- Binary hash: candidate `train_gpt2cu` sha256 `b388ebb5e16055d01b6a88f063ca8e67bf91b190ef716ee5751494cc658f6b96`; restored promoted-default `train_gpt2cu` sha256 `8a567cc6188709c1e0e68ece5cd0639c3baf0d51f35bb8fbb074bc5f39d6fe84`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2571.89, 2598.31, 2563.27 ms.
- Average: 2580.788732 ms.
- Default restore: rebuilt the promoted SM120 default binaries without `LLMK_SM120_CLASSIFIER_EXP2`; the restore build line retained `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, and `LLMK_SM120_USE_LIBTORCH_DRESIDUAL_ZERO`.
- Restore verification: the restored promoted smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject for promotion. This current-band run is `30.192 ms` faster than the slow maxconn1 telemetry reference `2610.981 ms`, but it is still `91.727 ms` slower than the promoted direct proof `2489.062 ms` and `87.656 ms` slower than the prior stable x10 baseline `2493.133 ms`, so it should not replace the promoted trainer or receive an x10 gate.

### `promoted_tk_dgelu_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_tk_dgelu_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_tk_dgelu_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_tk_dgelu_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: promoted SM120 fast default plus the TK fused dGELU dInput route: CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`.
- Binary hash: candidate `train_gpt2cu` sha256 `8d32b012b4cb1044a27347ba8a723bea51876315e0fde2957d924c60ad3e7274`; restored promoted-default `train_gpt2cu` sha256 `18743f0d703d3f0b569217176d6eb09ba9cec7ec61d2319a455421d912ff476d`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609932`, and training losses/norms matched the normal smoke band.
- Step timings: 2506.65, 2507.52, 2497.99 ms.
- Average: 2502.758503 ms.
- Default restore: rebuilt the promoted SM120 default binaries without `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`; the restore build line retained `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, and `LLMK_SM120_USE_LIBTORCH_DRESIDUAL_ZERO`.
- Restore verification: the restored promoted smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject for promotion. This current-band run is `108.223 ms` faster than the slow maxconn1 telemetry reference `2610.981 ms`, but it is still `13.696 ms` slower than the promoted direct proof `2489.062 ms` and `9.626 ms` slower than the prior stable x10 baseline `2493.133 ms`, so it should not replace the promoted trainer or receive an x10 gate.

### `promoted_tk_dgelu_classifier_exp2_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_tk_dgelu_classifier_exp2_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_tk_dgelu_classifier_exp2_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_tk_dgelu_classifier_exp2_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP -DLLMK_SM120_CLASSIFIER_EXP2" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: promoted SM120 fast default plus both recent current-band add-ons: CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`, and `LLMK_SM120_CLASSIFIER_EXP2`.
- Binary hash: candidate `train_gpt2cu` sha256 `0d54091aeaa825ef24a8112fa22c5a0bd0134307ca727d09d5ea5b62f8316c87`; restored promoted-default `train_gpt2cu` sha256 `ce7564e1a014a6c0ec4632e70e9a41fda0fa7b9a743885693cf7b9a26dc76f81`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609922`, and training losses/norms matched the normal smoke band.
- Step timings: 2622.91, 2607.32, 2612.42 ms.
- Average: 2609.867454 ms.
- Default restore: rebuilt the promoted SM120 default binaries without `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` or `LLMK_SM120_CLASSIFIER_EXP2`; the restore build line retained `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, and `LLMK_SM120_USE_LIBTORCH_DRESIDUAL_ZERO`.
- Restore verification: the restored promoted smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The two add-ons do not compose: the result is `107.109 ms` slower than TK dGELU alone, only `1.114 ms` faster than the slow maxconn1 telemetry reference `2610.981 ms`, and `120.805 ms` slower than the promoted direct proof `2489.062 ms`.

### `promoted_precompute_grad_scale_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_precompute_grad_scale_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_precompute_grad_scale_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_precompute_grad_scale_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: promoted SM120 fast default plus the precomputed AdamW grad-scale route: CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW`.
- Binary hash: candidate `train_gpt2cu` sha256 `758ce307d68ef4b5cc7bdb89323fa2265121630110cdc3e1fd481ba5877d9ec7`; restored promoted-default `train_gpt2cu` sha256 `4ea7b62c99eb08332b014aec754cdc7e85c6c1ac842826c4cf8d7371894893ca`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = precomputed device AdamW scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609930`, and training losses/norms matched the normal smoke band.
- Step timings: 2605.56, 2600.70, 2607.99 ms.
- Average: 2604.345322 ms.
- Default restore: rebuilt the promoted SM120 default binaries without `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW`; the restore build line retained `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, and `LLMK_SM120_USE_LIBTORCH_DRESIDUAL_ZERO`.
- Restore verification: the restored promoted smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The precomputed grad-scale route improves the current slow maxconn1 telemetry reference by only `6.636 ms`, remains `115.283 ms` slower than the promoted direct proof `2489.062 ms`, and is not a real training-speed promotion.

### `promoted_libtorch_grad_zero_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_libtorch_grad_zero_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_libtorch_grad_zero_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_libtorch_grad_zero_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=1 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: promoted SM120 fast default with LibTorch grad-zero replacing the promoted CUDA-kernel grad-zero route: Torch C++ grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Binary hash: candidate `train_gpt2cu` sha256 `f8d8d975ffa331f5f29fb67101887ea5968e3861f8ba75139f22303c83d37471`; restored promoted-default `train_gpt2cu` sha256 `f4fe6321c16b8dce8c58d68d3b391f340f221654d9f076798ee12856e3d7ed00`.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609911`, and training losses/norms matched the normal smoke band.
- Step timings: 2605.17, 2592.97, 2595.00 ms.
- Average: 2593.987226 ms.
- Default restore: rebuilt the promoted SM120 default binaries without `LLMK_SM120_USE_LIBTORCH_GRAD_ZERO`; the restore build line retained `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, and `LLMK_SM120_USE_LIBTORCH_DRESIDUAL_ZERO`.
- Restore verification: the restored promoted smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. LibTorch grad-zero improves the current slow maxconn1 telemetry reference by `16.994 ms`, but it remains `104.925 ms` slower than the promoted direct proof `2489.062 ms` and does not overturn the earlier LibTorch grad-zero stability rejection.

### `promoted_libtorch_grad_zero_tk_dgelu_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_libtorch_grad_zero_tk_dgelu_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_libtorch_grad_zero_tk_dgelu_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_libtorch_grad_zero_tk_dgelu_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=1 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: promoted SM120 fast default plus LibTorch grad-zero and the TK fused dGELU dInput route: Torch C++ grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`.
- Binary hash: candidate `train_gpt2cu` sha256 `c20620812ea1183f41a8fd7e3b1a90d678200a70277cf2235fed33fc8d6229ef`; restored promoted-default `train_gpt2cu` sha256 `f8a8bb9edaea5930096c3b45bd21a90e61cd03affa0f65f7658c5af728763535`.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Trainer correctness: initial val loss `11.033154`, final val loss `10.609930`, and training losses/norms matched the normal smoke band.
- Step timings: 2599.38, 2599.34, 2615.97 ms.
- Average: 2607.653975 ms.
- Default restore: rebuilt the promoted SM120 default binaries without `LLMK_SM120_USE_LIBTORCH_GRAD_ZERO` or `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`; the restore build line retained `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, and `LLMK_SM120_USE_LIBTORCH_DRESIDUAL_ZERO`.
- Restore verification: the restored promoted smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. LibTorch grad-zero plus TK fused dGELU does not compose positively: it is `104.895 ms` slower than TK dGELU alone, `13.667 ms` slower than LibTorch grad-zero alone, and `118.592 ms` slower than the promoted direct proof `2489.062 ms`.

### `promoted_cublas_dinp_tk_dgelu_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_tk_dgelu_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_cublas_dinp_tk_dgelu_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_cublas_dinp_tk_dgelu_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_ATTPROJ -DLLMK_SM120_USE_CUBLAS_DINP_FC -DLLMK_SM120_USE_TK_FUSED_DGELU_DINP" scripts/run_sm120_optimization_round.sh`.
- Build flags included the promoted default stack plus `LLMK_SM120_USE_CUBLAS_DINP_ATTPROJ`, `LLMK_SM120_USE_CUBLAS_DINP_FC`, and `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2487.63, 2490.46, 2492.12 ms.
- Average: 2491.288 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `463c358cd928c1fa1ecb60f56815f9a18043fa52a9c63ede819a5a5e156740ed`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: gate with x10. This was `1.845 ms` faster than stable x10 but `2.226 ms` slower than the promoted direct proof, so it required longer validation before any promotion.

### `promoted_cublas_dinp_tk_dgelu_x10_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_tk_dgelu_x10_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_cublas_dinp_tk_dgelu_x10_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_cublas_dinp_tk_dgelu_x10_20260521 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_ATTPROJ -DLLMK_SM120_USE_CUBLAS_DINP_FC -DLLMK_SM120_USE_TK_FUSED_DGELU_DINP" scripts/run_sm120_optimization_round.sh`.
- Build flags matched the x3 gate.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2506.66, 2504.58, 2493.06, 2514.62, 2521.94, 2515.46, 2514.69, 2503.64, 2503.50, 2504.61 ms.
- Average: 2508.456 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483711`.
- Binary hash: candidate `train_gpt2cu` sha256 `986e160c52d762ac91b720f9492bd2e9a9b59c89d9626b38479ff69fb9cd0062`; restored promoted-default sha256 `8254ee2d1065c6c921b2492f7349d6c3144982f2d7700b18e819a789cbd7c49e`.
- Restore verification: rebuilt the promoted SM120 default binaries without the rejected flags; the restored smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject. The short-run improvement did not survive the x10 gate; this composition is `19.394 ms` slower than the promoted direct proof and `15.323 ms` slower than stable x10.

### `promoted_memory_store_cg_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_memory_store_cg_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_memory_store_cg_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_memory_store_cg_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_MEMORY_STORE_POLICY=2" scripts/run_sm120_optimization_round.sh`.
- Build flags included the promoted default stack plus `LLMK_SM120_MEMORY_STORE_POLICY=2`, which changes the native memory wrapper stores from the promoted streaming-store policy to cache-global stores.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2596.26, 2588.53, 2596.98 ms.
- Average: 2592.754 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `c02b8a62028cb85f8880220918d3682ebc1798be0d5622f93aee878983445a52`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The candidate is `103.692 ms` slower than the promoted direct proof and `99.621 ms` slower than stable x10, so no x10 gate is justified.

### `promoted_memory_store_default_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_memory_store_default_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_memory_store_default_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_memory_store_default_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_MEMORY_STORE_POLICY=0" scripts/run_sm120_optimization_round.sh`.
- Build flags included the promoted default stack plus `LLMK_SM120_MEMORY_STORE_POLICY=0`, which changes the native memory wrapper stores from the promoted streaming-store policy to regular stores.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2600.82, 2591.76, 2594.85 ms.
- Average: 2593.304 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `961308d7118b107c34424009c5a673e590edfb1f1f9e03e4f445824522e38e9e`; restored promoted-default sha256 `b36d91d204f278eabdb8b5a3b69472d65552871e3f0b85926f37400df493efc9`.
- Restore verification: rebuilt the promoted SM120 default binaries without the rejected store-policy flags; the restored smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. Regular stores are `104.242 ms` slower than the promoted direct proof and `100.171 ms` slower than stable x10; keep the default `LLMK_SM120_MEMORY_STORE_POLICY=1` streaming-store route.

### `promoted_global_norm_block256_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_global_norm_block256_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_global_norm_block256_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_global_norm_block256_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_GLOBAL_NORM_BLOCK_SIZE=256" scripts/run_sm120_optimization_round.sh`.
- Build flags included the promoted default stack plus `LLMK_SM120_GLOBAL_NORM_BLOCK_SIZE=256`, retesting the earlier focused global-norm block-size signal in the actual promoted trainer.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2593.99, 2588.28, 2592.69 ms.
- Average: 2590.485573 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609930`.
- Binary hash: candidate `train_gpt2cu` sha256 `0a6ab68f0099b7c602ef65a46b8d2fdf9ee10348f36740b612b8239a9699fe53`; restored promoted-default `train_gpt2cu` sha256 `2a0804fdc9beaccaeb5fca5e02025963f919e1126be11f70f12fdf904c785a4f`.
- Restore verification: rebuilt the promoted SM120 default binaries without the rejected global-norm block override; the restored smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The 256-thread global-norm block is `101.423 ms` slower than the promoted direct proof and `97.353 ms` slower than stable x10. It improves the current slow telemetry band by `20.495 ms`, but that is runtime-drift-relative evidence only, not a faster trainer selection.

### `promoted_cublaslt_plan_cache_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_cublaslt_plan_cache_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_cublaslt_plan_cache_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_cublaslt_plan_cache_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_CACHE_CUBLASLT_PLANS" scripts/run_sm120_optimization_round.sh`.
- Build flags included the promoted default stack plus `LLMK_SM120_CACHE_CUBLASLT_PLANS`, retesting the earlier trainer-wide cuBLASLt plan-cache knob in the current promoted CUDA/Torch/maxconn composition.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2601.27, 2595.53, 2598.95 ms.
- Average: 2597.240567 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `67655be0c9df246fc5d0dde0157ed77c4895f7228c8111ccab6146c7086ca689`; restored promoted-default `train_gpt2cu` sha256 `d06861f636f10a165748cca4bed43e17fa8454ba826e7f66ddfa5d8973f53b73`.
- Restore verification: rebuilt the promoted SM120 default binaries without the rejected plan-cache flag; the restored smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. cuBLASLt plan caching is `108.178 ms` slower than the promoted direct proof and `104.108 ms` slower than stable x10. It improves the current slow telemetry band by only `13.740 ms`, so it is not a faster trainer selection.

### `promoted_cublaslt_ws256_heur16_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_cublaslt_ws256_heur16_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_cublaslt_ws256_heur16_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_cublaslt_ws256_heur16_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_CUBLASLT_WORKSPACE_MB=256 -DLLMK_SM120_CUBLASLT_HEURISTIC_RESULTS=16" scripts/run_sm120_optimization_round.sh`.
- Build flags included the promoted default stack plus the wider cuBLASLt workspace and heuristic-result search, retesting the earlier cuBLASLt tuning signal in the actual promoted CUDA/Torch/maxconn trainer composition.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2621.16, 2595.19, 2611.85 ms.
- Average: 2603.521466 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `6dd93ebab9fa05243102d684fc8fc2e3675c8343ac2197e61b7278054e16495a`; restored promoted-default `train_gpt2cu` sha256 `a55413ce56de718c387d0b21744c7f3a627a80f2c048da7a092b84e49c79545d`.
- Restore verification: rebuilt the promoted SM120 default binaries without the rejected cuBLASLt workspace/search override; the restored smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The wider cuBLASLt workspace/search candidate is `114.459 ms` slower than the promoted direct proof and `110.388 ms` slower than stable x10. It improves the current slow telemetry band by only `7.460 ms`, so it is not a faster trainer selection.

### `promoted_cublas_dinp_qkv_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_qkv_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_cublas_dinp_qkv_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_cublas_dinp_qkv_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_QKV" scripts/run_sm120_optimization_round.sh`.
- Build flags included the promoted default stack plus the qkv-only direct-cuBLAS dInput selector, retesting that earlier selector in the actual promoted CUDA/Torch/maxconn trainer composition.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2611.45, 2606.87, 2607.71 ms.
- Average: 2607.288361 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `a177b778de5343db8be9c12959bba277a4875dd5d708e6f64ea8de78eb31139d`; restored promoted-default `train_gpt2cu` sha256 `3a268f8f03fd7379b21fb2b11c0314cab543dd839fe01b1f102bc783ce47e4cc`.
- Restore verification: rebuilt the promoted SM120 default binaries without the rejected qkv direct-cuBLAS selector; the restored smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The qkv direct-cuBLAS dInput selector is `118.226 ms` slower than the promoted direct proof and `114.155 ms` slower than stable x10. It improves the current slow telemetry band by only `3.693 ms`, so it is not a faster trainer selection.

### `promoted_cublas_dinp_fcproj_x3_20260521`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_fcproj_x3_20260521`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_cublas_dinp_fcproj_x3_20260521`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_cublas_dinp_fcproj_x3_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_FCPROJ" scripts/run_sm120_optimization_round.sh`.
- Build flags included the promoted default stack plus the fcproj-only direct-cuBLAS dInput selector, retesting the old fcproj selector in the actual promoted CUDA/Torch/maxconn trainer composition.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2504.18, 2498.36, 2489.46 ms.
- Average: 2493.910909 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `b19a5f6214ffe09c4e94f6f783f1b915f4f7f9589e42eda3a1a6706808357f65`; restored promoted-default `train_gpt2cu` sha256 `6c6f8a1495c19f010fe320d7adebad17ea127ea8535e0ef9e4d9a7c5c87755bf`.
- Restore verification: rebuilt the promoted SM120 default binaries without the rejected fcproj direct-cuBLAS selector; the restored smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject as a near miss. The fcproj direct-cuBLAS dInput selector is `4.849 ms` slower than the promoted direct proof and `0.778 ms` slower than stable x10, so it is not a faster trainer selection even though it is `117.070 ms` faster than the current slow telemetry band.

### `promoted_cublas_dinp_fcproj_x10_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_fcproj_x10_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_cublas_dinp_fcproj_x10_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_cublas_dinp_fcproj_x10_20260522 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_FCPROJ" scripts/run_sm120_optimization_round.sh`.
- Build flags included the promoted default stack plus the fcproj-only direct-cuBLAS dInput selector.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2485.23, 2479.73, 2484.31, 2488.62, 2489.16, 2492.40, 2491.63, 2494.12, 2494.32, 2495.89 ms.
- Average: 2490.019348 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: candidate `train_gpt2cu` sha256 `8c11daffd3d3d0f97b3e72838964fe4fead484ad5e527dff68aa70bc46812662`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: gate for confirmation but do not promote from this run alone. It beat stable x10 by `3.114 ms` and the same-session promoted default by `43.465 ms`, but it was still `0.957 ms` slower than promoted direct `train-sm120.sh` evidence.

### `promoted_default_same_session_x10_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_default_same_session_x10_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_default_same_session_x10_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_default_same_session_x10_20260522 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Build flags matched the promoted default stack without extra candidate flags.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2490.68, 2489.65, 2580.94, 2553.13, 2495.74, 2525.01, 2564.39, 2563.42, 2521.29, 2507.79 ms.
- Average: 2533.483876 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: promoted-default `train_gpt2cu` sha256 `a6ac04586625fa772ee8d052215717d6ddc1c8d05169ab1c4e16e8e33b305694`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: keep as a regression control. The source-selected stack did not change, but this run shows current runtime drift and late-step spikes can make the promoted default slower than its direct proof.

### `promoted_cublas_dinp_fcproj_confirm_x10_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_fcproj_confirm_x10_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_cublas_dinp_fcproj_confirm_x10_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_cublas_dinp_fcproj_confirm_x10_20260522 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_FCPROJ" scripts/run_sm120_optimization_round.sh`.
- Build flags matched the first fcproj x10 candidate.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2486.81, 2482.43, 2484.14, 2485.07, 2518.52, 2614.79, 2583.46, 2565.23, 2598.75, 2600.12 ms.
- Average: 2548.056417 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: candidate `train_gpt2cu` sha256 `6be5f2aded0df76a7083c5ba918c05bb31e444b1310caa19a677ee4e9328abe7`; restored promoted-default `train_gpt2cu` sha256 `1edf1131bcc110f3b2cc16a4a090273f4920807f661547e864877f901c236c85`.
- Restore verification: rebuilt the promoted SM120 default binaries without the rejected fcproj selector; the restored smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject as unstable after confirmation. The candidate failed the repeat x10 with late-step spikes and is `58.994 ms` slower than promoted direct proof, `54.923 ms` slower than stable x10, and `14.573 ms` slower than the same-session promoted default.

### `promoted_async_grad_norm_copy_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_async_grad_norm_copy_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_async_grad_norm_copy_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_async_grad_norm_copy_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_ASYNC_GRAD_NORM_COPY" scripts/run_sm120_optimization_round.sh`.
- Build flags included the promoted default stack plus `LLMK_SM120_ASYNC_GRAD_NORM_COPY`, which changes the grad-norm scalar readback to a stream-scoped async copy plus stream synchronize.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2483.38, 2526.11, 2490.07 ms.
- Average: 2508.090854 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `a0e82a97006ae371b599ba6eea57fb1c01f7fcd876b860100307efbf05e7fd98`; restored promoted-default `train_gpt2cu` sha256 `30e73fb995fbca0d18cdb0ec976da12908ac2cd34939cf3c81e9aafb3d1b31aa`.
- Restore verification: rebuilt the promoted SM120 default binaries without the rejected async grad-norm-copy flag; the restored smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. This promoted-stack retest is `19.029 ms` slower than promoted direct proof and `14.958 ms` slower than stable x10, so the async scalar-copy route does not improve the trainer.

### `promoted_device_grad_scale_adamw_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_device_grad_scale_adamw_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_device_grad_scale_adamw_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_device_grad_scale_adamw_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DEVICE_GRAD_SCALE_ADAMW" scripts/run_sm120_optimization_round.sh`.
- Build flags included the promoted default stack plus `LLMK_SM120_DEVICE_GRAD_SCALE_ADAMW`, which lets AdamW compute `grad_scale` from the device-resident grad-norm-squared scalar.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = device AdamW scalar`, and `gelu_fusion = 1`.
- Step timings: 2521.80, 2538.11, 2518.78 ms.
- Average: 2528.445721 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609930`.
- Binary hash: candidate `train_gpt2cu` sha256 `636da218083f11c6a79855e7e7a349c27c7e1b605ae27eda7c506889933ec9b6`; restored promoted-default `train_gpt2cu` sha256 `68e1745626dc3cdf2e6ccc825b8dd5fed70bb6ff7601c1f1730aacfcf07f0fc9`.
- Restore verification: rebuilt the promoted SM120 default binaries without the rejected device-grad-scale flag; the restored smoke suite passed `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The device AdamW scalar route is `39.384 ms` slower than promoted direct proof and `35.313 ms` slower than stable x10. The already-rejected precomputed scalar route remains the better device-grad-scale variant, and neither beats the selected host-scalar AdamW route.

### `promoted_maxconn4_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_maxconn4_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_maxconn4_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=4 RUN_LABEL=codex_sm120_promoted_maxconn4_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Build flags used the promoted default stack; this was a runtime scheduling-only retest of `CUDA_DEVICE_MAX_CONNECTIONS=4`.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2485.71, 2504.03, 2559.56 ms.
- Average: 2531.797528 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: promoted-default `train_gpt2cu` sha256 `3a7b18eb7720a58263e5f152f516e31bf3f9d9f0c7aa63450b175818ff07c48f`.
- Restore verification: not required because the candidate changed only the runtime `CUDA_DEVICE_MAX_CONNECTIONS` setting and the manifest hash matches the current `train_gpt2cu`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. `CUDA_DEVICE_MAX_CONNECTIONS=4` is faster than the already-rejected maxconn2 x3, but it is still `42.735 ms` slower than promoted direct proof and `38.665 ms` slower than stable x10. Keep the promoted `train-sm120.sh` maxconn1 runtime setting.

### `promoted_maxconn8_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_maxconn8_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_maxconn8_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=8 RUN_LABEL=codex_sm120_promoted_maxconn8_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Build flags used the promoted default stack; this was a runtime scheduling-only retest of `CUDA_DEVICE_MAX_CONNECTIONS=8`.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2487.65, 2480.71, 2486.20 ms.
- Average: 2483.451128 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `da37dd766e57ec27d0cd08414f7d2a27ee20ec2c2ed5077231c517612528b78d`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: x10 gate, then reject after the longer gate. The short run was `5.611 ms` faster than promoted direct proof and `9.682 ms` faster than stable x10, but prior short-run scheduling wins were not trusted without stability proof.

### `promoted_maxconn8_x10_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_maxconn8_x10_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_maxconn8_x10_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=8 RUN_LABEL=codex_sm120_promoted_maxconn8_x10_20260522 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: same promoted default stack as the x3 gate, with `CUDA_DEVICE_MAX_CONNECTIONS=8`.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2516.38, 2509.12, 2487.32, 2522.68, 2534.17, 2492.36, 2492.93, 2494.32, 2538.47, 2518.16 ms.
- Average: 2509.946770 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: candidate `train_gpt2cu` sha256 `6d2b3a551cb1ec8e344668dedfbcfac11e0b7882184fa5ab28621ddcec36f707`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject. The x10 gate did not preserve the x3 speed: it is `20.885 ms` slower than promoted direct proof and `16.814 ms` slower than stable x10. Keep `train-sm120.sh` on `CUDA_DEVICE_MAX_CONNECTIONS=1`.

### `promoted_maxconn16_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_maxconn16_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_maxconn16_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=16 RUN_LABEL=codex_sm120_promoted_maxconn16_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Build flags used the promoted default stack; this was a runtime scheduling-only retest of `CUDA_DEVICE_MAX_CONNECTIONS=16`.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2485.76, 2484.84, 2485.39 ms.
- Average: 2485.113382 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `83bc22c45f86b02d2eded37a23458eef6950a6b9b35074e6ae5a54f9f4779aed`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: x10 gate, then reject after the longer gate. The short run was `3.949 ms` faster than promoted direct proof and `8.020 ms` faster than stable x10, but scheduler-only wins require a longer stability run before any `train-sm120.sh` change.

### `promoted_maxconn16_x10_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_maxconn16_x10_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_maxconn16_x10_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=16 RUN_LABEL=codex_sm120_promoted_maxconn16_x10_20260522 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: same promoted default stack as the x3 gate, with `CUDA_DEVICE_MAX_CONNECTIONS=16`.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2604.76, 2577.37, 2542.35, 2507.44, 2531.21, 2523.16, 2504.25, 2506.08, 2523.51, 2518.24 ms.
- Average: 2525.956233 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: candidate/current `train_gpt2cu` sha256 `f6e6aa878a03a8a9f23c52aa3e2f3c63408ab881f9a73a00b7e4223e1f7680e0`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: reject. The x10 gate did not preserve the x3 speed: it is `36.894 ms` slower than promoted direct proof and `32.823 ms` slower than stable x10. Keep `train-sm120.sh` on `CUDA_DEVICE_MAX_CONNECTIONS=1`.

### `promoted_default_post_sched_x10_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_default_post_sched_x10_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_default_post_sched_x10_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_default_post_sched_x10_20260522 MAX_STEPS=10 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: promoted default stack with the selected runtime scheduling setting, matching `train-sm120.sh`.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2527.77, 2592.46, 2595.51, 2533.60, 2489.10, 2489.09, 2494.04, 2501.22, 2609.20, 2586.29 ms.
- Average: 2543.389850 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: promoted-default `train_gpt2cu` sha256 `d1db5331be49543b6a7388c86d95f99976d1591d5798a9885ee053235c0cc858`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000010` and `main.log`.
- Decision: regression control. The selected stack itself is currently running `54.328 ms` slower than the promoted direct proof and `50.257 ms` slower than stable x10, so the slower `train-sm120.sh` band is not explained by accidentally keeping maxconn8/maxconn16 or another rejected candidate enabled.

### `promoted_attn_fwd16_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_attn_fwd16_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_attn_fwd16_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_attn_fwd16_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_ATTN_FWD_BLOCK=16" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: promoted default stack with the forward attention tile reduced from the selected 32 rows to 16 rows; backward stayed on the selected 16-row tile.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Native benchmark evidence: `bench_sm120_attention.log` measured attention forward/backward at `1055.652/2747.087 us`; the selected packed-QKV route in `codex_sm120_round_backward_stream_sync_default_x10_20260521` measured forward/backward at `784.691/2716.901 us`, so the forward-tile change worsens the focused row.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2534.95, 2526.02, 2536.46 ms.
- Average: 2531.237483 ms.
- Losses: initial val loss `11.033163`, final val loss `10.609920`.
- Binary hash: candidate `train_gpt2cu` sha256 `281dc5e6cd37cdb5c9b06b024e1513e38312877f2a75bf823e285eaf8b37821c`; restored promoted-default `train_gpt2cu` sha256 `930321ac2faefca3e9300795756cd37b831923885ed36b551a181e51b2450ff2`.
- Restore verification: rebuilt the promoted default stack without `LLMK_SM120_ATTN_FWD_BLOCK=16`; the restore run `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_attn_fwd16_20260522` passed all nine focused CUDA smokes.
- Cleanup: harness removed candidate checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The candidate improved over the latest slow selected-stack x10 control by `12.152 ms`, but it is still `42.175 ms` slower than promoted direct proof and `38.104 ms` slower than stable x10. Do not x10-gate or promote.

### `promoted_attn_bwd32_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_attn_bwd32_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_attn_bwd32_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_ATTN_BWD_BLOCK=32" scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: promoted default stack with the backward attention tile increased from the selected 16 rows to 32 rows; forward stayed on the selected 32-row tile.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Native benchmark evidence: `bench_sm120_attention.log` measured attention forward/backward at `830.444/3074.490 us`; the selected packed-QKV route in `codex_sm120_round_backward_stream_sync_default_x10_20260521` measured forward/backward at `784.691/2716.901 us`, so the backward-tile change worsens both focused rows, especially backward.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2511.08, 2502.21, 2502.79 ms.
- Average: 2502.500892 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `d254931734f4b26360e258efba5f65c1ffe5929fefc2d70d3ea567e5053cf8be`; restored promoted-default `train_gpt2cu` sha256 `9940b7593e8ecf0e787da28ba17c3fd3c6eae5a431124312daf94d2383a323fe`.
- Restore verification: rebuilt the promoted default stack without `LLMK_SM120_ATTN_BWD_BLOCK=32`; the restore run `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_attn_bwd32_20260522` passed all nine focused CUDA smokes.
- Cleanup: harness removed candidate checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The candidate improved over the latest slow selected-stack x10 control by `40.889 ms`, but it is still `13.439 ms` slower than promoted direct proof and `9.368 ms` slower than stable x10, while the attention microbenchmark worsened. Do not x10-gate or promote.

### `cublaslt_epilogue_probe_20260522`

- Artifact file: `scratch/sm120_rounds/codex_sm120_cublaslt_epilogue_probe_20260522/bench_sm120_cublaslt_epilogue_algos.log`
- Command: `LLMK_BENCH_REPEATS=5 LLMK_BENCH_ITERS=6 ./bench_sm120_cublaslt_epilogue_algos`.
- Build/runtime settings: focused cuBLASLt fused-epilogue algorithm probe on the current SM120 binaries; no trainer candidate flags.
- Benchmark evidence: `fc fwd+GeLU` returned two heuristics and selected default index 0; index 0 measured `1512.053 us`, index 1 measured `1557.621 us`.
- Benchmark evidence: `fcproj dInp+dGeLU` returned two heuristics; index 1 measured `1901.317 us`, but a default-index retest measured index 0 at `1871.051 us`.
- Training evidence: not run. The focused same-shape probe did not show a stable alternate-algorithm edge, so a global `LLMK_SM120_CUBLASLT_HEURISTIC_INDEX=1` trainer run would not be benchmark-backed.
- Decision: benchmark-reject. Keep the default lowest-waves cuBLASLt selector for the promoted stack and do not x3-gate a heuristic-index candidate from this probe.

### `promoted_default_after_epilogue_probe_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_default_after_epilogue_probe_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_default_after_epilogue_probe_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_default_after_epilogue_probe_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Build/runtime settings: promoted SM120 default stack with no extra candidate flags.
- Focused correctness: all nine focused CUDA smokes passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2600.25, 2592.99, 2596.78 ms.
- Average: 2594.887972 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: current promoted-default `train_gpt2cu` sha256 `304550070058a3abe5d455d248688a586d923321115fcadeb10be072b2e9cf04`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: regression control. The selected stack is still running in a slow current runtime band, `105.826 ms` slower than promoted direct proof and `101.755 ms` slower than stable x10. This does not redefine the promotion baseline and does not justify changing defaults.

### `direct_train_sm120_current_no_rebuild_x10_20260522`

- Artifact file: `scratch/sm120_rounds/codex_sm120_direct_train_sm120_current_no_rebuild_x10_20260522/train-sm120.log`
- Trainer output: `log124M/5090_S`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh`.
- Build/runtime settings: current promoted SM120 default stack via the user-facing script, with no rebuild or candidate flags in this run.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1` on `NVIDIA GeForce RTX 5090`.
- Step timings: 2573.25, 2549.62, 2575.55, 2557.87, 2612.06, 2541.98, 2523.45, 2514.89, 2500.04, 2499.23 ms.
- Average: 2541.632679 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Cleanup: removed the generated `log124M/5090_S/model_00000010.bin` and `log124M/5090_S/state_00000010_00000.bin` checkpoint files after the run.
- Decision: regression control. This exact script path is `52.571 ms` slower than the promoted direct proof and `48.500 ms` slower than stable x10 even though the selected backends are active, so the slowdown is real but is not evidence that a rejected candidate flag was left enabled.

### `promoted_cublaslt_heur1_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_cublaslt_heur1_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_cublaslt_heur1_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_cublaslt_heur1_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_CUBLASLT_HEURISTIC_RESULTS=1" scripts/run_sm120_optimization_round.sh`.
- Rationale: test whether the current promoted path slowed down because the wider default cuBLASLt heuristic search (`LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS=8`) changed selected algorithms versus a first-heuristic route.
- Build/runtime settings: promoted SM120 default stack plus forced one-result cuBLASLt heuristic search: CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark evidence: `bench_sm120_matmul` included qkv `fwd/dInp/dW/dW+accum` cuBLASLt timings `1101.30/1007.25/1019.21/992.93 us`, fc `fwd+GeLU/dInp/dW/dW+accum` cuBLASLt timings `1442.89/1373.61/1321.46/1331.39 us`, and fcproj `fwd/dInp/dInp+dGeLU/dW/dW+accum` cuBLASLt timings `1480.97/1380.01/1802.48/1313.26/1317.40 us`.
- Other focused benchmarks: attention forward/backward `782.982/2745.302 us`; LayerNorm C=768 forward/fused/backward `139.876/279.727/277.450 us`; CUDA grad memset row `157.282 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2553.50, 2542.25, 2563.48 ms.
- Average: 2552.864432 ms.
- Losses: initial val loss `11.033152`, final val loss `10.609923`.
- Binary hash: candidate `train_gpt2cu` sha256 `7fabceccbbbd0ee79e3cb4db71f78d85f25adf62c299b1a5e7ee54d1888b0c1f`; restored promoted-default `train_gpt2cu` sha256 `240bb78c20ce1a3d1754ce2690350351e2dc6ebc79325f1a6b154ee002f4f278`.
- Restore verification: rebuilt the promoted default stack without `LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS=1` in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_heur1_20260522`; all nine focused CUDA smokes passed.
- Cleanup: harness removed candidate checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The one-heuristic cuBLASLt route is `63.802 ms` slower than promoted direct proof, `59.731 ms` slower than stable x10, and `11.232 ms` slower than the current direct-script regression control. It does not explain or recover the promoted-stack slowdown and should not replace the default heuristic selector.

### `promoted_disable_cublas_bwd_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_disable_cublas_bwd_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_disable_cublas_bwd_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_disable_cublas_bwd_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM" scripts/run_sm120_optimization_round.sh`.
- Rationale: current focused rows showed several dWeight/dInput cases where cuBLASLt was competitive with or better than the older direct-cuBLAS backward selector, so this tested whether disabling the selector recovers end-to-end speed.
- Build/runtime settings: promoted SM120 default stack plus disabled direct-cuBLAS backward GEMM selector; CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark evidence: `bench_sm120_matmul` showed qkv `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `1011.54/1102.30/1140.00 us` versus `1007.37/1010.96/996.27 us`; attproj `365.69/372.40/379.79 us` versus `363.38/326.93/331.24 us`; fc `1350.52/1493.96/1537.65 us` versus `1335.16/1333.90/1313.67 us`; fcproj `1392.16/1470.99/1474.77 us` versus `1403.16/1354.08/1318.61 us`; and lmhead `21851.27/21040.21/21064.47 us` versus `21407.71/21186.74/21368.90 us`.
- Other focused benchmarks: attention forward/backward `782.906/2748.105 us`; LayerNorm C=768 forward/fused/backward `137.188/278.531/274.538 us`; CUDA grad memset row `159.226 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2527.46, 2521.17, 2521.26 ms.
- Average: 2521.216512 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609921`.
- Binary hash: candidate `train_gpt2cu` sha256 `c4422f57aca3ef214bc43d839f15150bb0b06a6c2150f5a347958b430e55cfdf`; restored promoted-default `train_gpt2cu` sha256 `da47e81e5a54dfa3ae1b1cbe3d5fc052ea94b84d9ad4216e466ff4c420edc0ff`.
- Restore verification: rebuilt the promoted default stack without `LLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM` in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_disable_cublas_bwd_20260522`; all nine focused CUDA smokes passed.
- Cleanup: harness removed candidate checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject-current-band-win. Disabling the direct-cuBLAS backward selector is `20.416 ms` faster than the current direct-script regression control and `73.671 ms` faster than the latest x3 promoted-stack regression control, but it remains `32.154 ms` slower than promoted direct proof and `28.084 ms` slower than stable x10. This is useful regression evidence, not a training-speed promotion.

### `disable_cublas_bwd_maxconn8_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_disable_cublas_bwd_maxconn8_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_disable_cublas_bwd_maxconn8_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=8 RUN_LABEL=codex_sm120_disable_cublas_bwd_maxconn8_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM" scripts/run_sm120_optimization_round.sh`.
- Rationale: compose the current-band direct-cuBLAS-backward-disable recovery with a wider CUDA work-queue setting to see whether the scheduling knob helped only the promoted default stack or also the backward-selector candidate.
- Build/runtime settings: promoted SM120 default stack plus disabled direct-cuBLAS backward GEMM selector and `CUDA_DEVICE_MAX_CONNECTIONS=8`; CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark evidence: not run for this composed scheduling check; correctness plus TinyStories timing were the gate.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2537.44, 2534.58, 2534.18 ms.
- Average: 2534.379601 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609921`.
- Binary hash: candidate `train_gpt2cu` sha256 `3839f75c3d9c3427c4c1db075af41af009660f01a231bffdb264f78b0ed8c83a`; restored promoted-default `train_gpt2cu` sha256 `0d638c6cd459e201ef06a0f8854c3d51e96c5769a67570c995c8cd209a89cf66`.
- Restore verification: rebuilt the promoted default stack without `LLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM` in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_disable_cublas_bwd_maxconn8_20260522`; all nine focused CUDA smokes passed.
- Cleanup: harness removed candidate checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject-current-band-partial. This composed route is `7.253 ms` faster than the current direct-script regression control, but it is `13.163 ms` slower than the same disabled-backward-selector candidate at `CUDA_DEVICE_MAX_CONNECTIONS=1`, `45.317 ms` slower than promoted direct proof, and `41.247 ms` slower than stable x10. Keep the promoted default stack on `CUDA_DEVICE_MAX_CONNECTIONS=1`.

### `disable_cublas_bwd_attn_bwd32_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_disable_cublas_bwd_attn_bwd32_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_disable_cublas_bwd_attn_bwd32_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_disable_cublas_bwd_attn_bwd32_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM -DLLMK_SM120_ATTN_BWD_BLOCK=32" scripts/run_sm120_optimization_round.sh`.
- Rationale: compose the two independent current-band recovery signals from the disabled direct-cuBLAS backward selector and the attention backward block-32 override to test whether their gains stack in the real trainer.
- Build/runtime settings: promoted SM120 default stack plus disabled direct-cuBLAS backward GEMM selector and `LLMK_SM120_ATTN_BWD_BLOCK=32`; CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark evidence: qkv `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `1012.89/1108.11/1110.87 us` versus `1007.21/1017.44/1001.42 us`; fc `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `1386.04/1521.26/1494.37 us` versus `1373.71/1308.32/1309.63 us`; fcproj `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `1452.49/1469.48/1516.29 us` versus `1365.65/1326.44/1312.59 us`; attention forward/backward `786.227/2872.892 us`; LayerNorm C=768 forward/fused/backward `141.088/279.358/277.084 us`; CUDA grad memset row `175.536 us`; AdamW update row `1913.203 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2546.80, 2569.27, 2538.59 ms.
- Average: 2553.930163 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609921`.
- Binary hash: candidate `train_gpt2cu` sha256 `29a84804f143a9d01eb3a64d0d6afe50bd2e5976990164c0a5ecdd856f514d8f`; restored promoted-default `train_gpt2cu` sha256 `267fdc471f8a924751e551ed09bc98811092e0b3a7865d05a866ef1f4443f35a`.
- Restore verification: rebuilt the promoted default stack without `LLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM` or `LLMK_SM120_ATTN_BWD_BLOCK=32` in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_disable_cublas_bwd_attn_bwd32_20260522`; all nine focused CUDA smokes passed.
- Cleanup: harness removed candidate checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The two current-band recovery signals compose negatively: this route is `32.714 ms` slower than the disabled-backward-selector candidate alone, `51.429 ms` slower than the attention-bwd32 candidate alone, `12.297 ms` slower than the current direct-script regression control, `64.868 ms` slower than promoted direct proof, and `60.797 ms` slower than stable x10. Do not x10-gate or promote.

### `disable_cublas_bwd_tk_dgelu_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_disable_cublas_bwd_tk_dgelu_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_disable_cublas_bwd_tk_dgelu_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_disable_cublas_bwd_tk_dgelu_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM -DLLMK_SM120_USE_TK_FUSED_DGELU_DINP" scripts/run_sm120_optimization_round.sh`.
- Rationale: compose the current-band disabled direct-cuBLAS backward selector with the trainer-callable TK fused dGELU dInput route. This differs from the earlier direct-cuBLAS dInput plus TK dGELU composition and tests whether the broader backward-selector change interacts better with the TK fused dGELU route.
- Build/runtime settings: promoted SM120 default stack plus disabled direct-cuBLAS backward GEMM selector and `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`; CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark evidence: qkv `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `1028.82/1108.15/1135.13 us` versus `1011.47/1007.16/990.49 us`; fc `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `1379.70/1520.74/1497.46 us` versus `1385.66/1311.29/1312.33 us`; fcproj `dInp+dGeLU` TK/cuBLASLt-fused timings `1796.61/1817.67 us`; attention forward/backward `787.623/2742.551 us`; LayerNorm C=768 forward/fused/backward `142.421/280.343/273.218 us`; CUDA grad memset row `157.758 us`; AdamW update row `1813.523 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2535.74, 2541.56, 2532.31 ms.
- Average: 2536.936402 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609922`.
- Binary hash: candidate `train_gpt2cu` sha256 `64551d3f333722d8bda80fcc900a70c330a8fbdadc9a40d34639f32e7eaa178e`; restored promoted-default `train_gpt2cu` sha256 `1d89cc669368b2d163f3f0edf994ead1aeab075b833b056d4f99de4b18a33d37`.
- Restore verification: rebuilt the promoted default stack without `LLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM` or `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_disable_cublas_bwd_tk_dgelu_20260522`; all nine focused CUDA smokes passed.
- Cleanup: harness removed candidate checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject-current-band-partial. This composition is `4.696 ms` faster than the current direct-script regression control, but it is `15.720 ms` slower than the disabled-backward-selector candidate alone, `34.178 ms` slower than the promoted TK dGELU candidate alone, `47.874 ms` slower than promoted direct proof, and `43.803 ms` slower than stable x10. The focused `dInp+dGeLU` row win does not translate into a trainer win in this composition, so do not x10-gate or promote.

### `cublas_dinp_fcproj_maxconn8_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_maxconn8_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_cublas_dinp_fcproj_maxconn8_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=8 RUN_LABEL=codex_sm120_cublas_dinp_fcproj_maxconn8_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_FCPROJ" scripts/run_sm120_optimization_round.sh`.
- Rationale: compose the previously near-miss fcproj-only direct-cuBLAS dInput selector with the fastest short-run scheduler signal, `CUDA_DEVICE_MAX_CONNECTIONS=8`, to check whether the two independent x3 signals stack in the real trainer.
- Build/runtime settings: promoted SM120 default stack plus `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` and `CUDA_DEVICE_MAX_CONNECTIONS=8`; CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark evidence: qkv `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `1088.98/1174.89/1179.28 us` versus `1071.38/1033.77/1085.28 us`; attproj `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `407.93/395.49/397.50 us` versus `388.80/347.35/352.79 us`; fc `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `1461.98/1558.17/1601.35 us` versus `1417.64/1395.63/1372.30 us`; fcproj `dInp` TK/cuBLASLt/cuBLAS timings `1579.87/1483.60/1534.85 us`; fcproj `dInp+dGeLU` TK/cuBLASLt-fused/cuBLASLt-explicit/cuBLAS-explicit timings `1861.89/1937.61/2335.38/2307.52 us`; attention forward/backward `824.824/2913.986 us`; LayerNorm C=768 forward/fused/backward `149.741/300.240/289.526 us`; CUDA grad memset row `171.456 us`; AdamW update row `1948.336 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2607.28, 2652.98, 2703.26 ms.
- Average: 2678.119063 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `4cb18feeeab14db071c355f9429079f0f485e7e935c57b9b339bd7a47babbea9`; restored promoted-default `train_gpt2cu` sha256 `960c820c780318f9e21e568d7fac227de02e3a9e65877f03e1edb996ca8415e4`.
- Restore verification: rebuilt the promoted default stack without `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` and with `CUDA_DEVICE_MAX_CONNECTIONS=1` in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_cublas_dinp_fcproj_maxconn8_20260522`; all nine focused CUDA smokes passed.
- Cleanup: harness removed candidate checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject. The row-level fcproj direct-cuBLAS and short-run maxconn8 signals compose badly in the trainer: this route is `184.208 ms` slower than the prior fcproj-only x3 candidate, `194.668 ms` slower than maxconn8 alone, `136.486 ms` slower than the current direct-script regression control, `189.057 ms` slower than promoted direct proof, and `184.986 ms` slower than stable x10. Do not x10-gate or promote.

### `promoted_disable_backward_stream_sync_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_disable_backward_stream_sync_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_disable_backward_stream_sync_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DISABLE_BACKWARD_STREAM_SYNC" scripts/run_sm120_optimization_round.sh`.
- Rationale: test whether the current slow runtime band is tied to the promoted stream-scoped backward synchronization by forcing the older device-wide synchronization path while keeping the rest of the promoted stack unchanged.
- Build/runtime settings: promoted SM120 default stack plus `LLMK_SM120_DISABLE_BACKWARD_STREAM_SYNC`; CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark evidence: qkv `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `1013.56/1112.44/1111.36 us` versus `1007.41/998.16/994.34 us`; attproj `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `371.93/372.03/375.30 us` versus `364.00/327.88/329.61 us`; fc `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `1441.92/1533.48/1481.40 us` versus `1458.61/1330.15/1309.92 us`; fcproj `dInp` TK/cuBLASLt/cuBLAS timings `1691.05/1536.44/1575.39 us`; fcproj `dInp+dGeLU` TK/cuBLASLt-fused/cuBLASLt-explicit/cuBLAS-explicit timings `1991.25/2130.47/2561.87/2411.68 us`; attention forward/backward `797.471/2848.220 us`; LayerNorm C=768 forward/fused/backward `141.178/284.532/275.974 us`; CUDA grad memset row `163.107 us`; AdamW update row `1880.858 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2589.33, 2551.94, 2518.34 ms.
- Average: 2535.140514 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `02132d4e76f982f2e7112e870526c99321ae75993414964c589648644f617bc6`; restored promoted-default `train_gpt2cu` sha256 `c8934ab48aa399d4dfa218bd50f91768c526e3931d4226d95cf35764977f7d2c`.
- Restore verification: rebuilt the promoted default stack without `LLMK_SM120_DISABLE_BACKWARD_STREAM_SYNC` in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_disable_backward_stream_sync_20260522`; all nine focused CUDA smokes passed.
- Cleanup: harness removed candidate checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: reject-current-band-partial. The older device-wide synchronization path is `6.492 ms` faster than the current direct-script regression control, but it remains `46.078 ms` slower than promoted direct proof, `42.008 ms` slower than stable x10, and far behind the original stream-sync x3 evidence. Keep the promoted stream-scoped backward synchronization.

### `antigravity_binary_probe_x3_20260522`

- Trainer output: `log124M/5090_S_antigravity_probe_x3_20260522`
- Command: `./train_gpt2cu_antigravity -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_antigravity_probe_x3_20260522 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`.
- Binary sha256: `948b57bb3b5cd55792f8e8711c48f849ba5007e0ee47ce3c3ceef12fb181e25a`.
- Rationale: test whether the separate older binary explains the faster `new-goal.md` historical target before blaming the current source-selected stack.
- Startup does not print the newer backend selector lines, so this is a diagnostic binary reference rather than a source-promotable kernel combination.
- Step timings: 3304.33, 3309.41, 3315.87 ms.
- Average: 3312.639713 ms.
- Losses: initial val loss `11.033152`, final val loss `10.609902`.
- Decision: rejected. This is `842.797 ms` slower than the `new-goal.md` target, `823.578 ms` slower than promoted direct proof, `819.507 ms` slower than stable x10, and `771.007 ms` slower than the current direct-script regression control.
- Cleanup: removed `model_00000003.bin` and `state_00000003_00000.bin`; retained `main.log` and `DONE_00000003`.

### `current_selected_runtime_telemetry_x3_20260522`

- Trainer output: `log124M/5090_S_runtime_telemetry_probe_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_runtime_telemetry_probe_x3_20260522 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`.
- Rationale: recheck whether the current selected stack is intrinsically slow while recording GPU runtime telemetry.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1` on `NVIDIA GeForce RTX 5090`.
- Telemetry: dmon samples during the training phase showed `99-100%` SM utilization, `574-575 W`, graphics clocks around `2692-2775 MHz`, memory clock `13801 MHz`, GPU temperature rising from roughly `45 C` to `55 C`, and `pviol=0` / `tviol=0`.
- Step timings: 2480.70, 2476.69, 2482.10 ms.
- Average: 2479.393840 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Decision: recovered-band control. The selected stack can still run fast and this short run is `62.239 ms` faster than the degraded direct-script control and `9.668 ms` faster than promoted direct proof. It remains `9.551 ms` slower than the `new-goal.md` first-three target, so it is not goal-complete evidence.
- Cleanup: removed `model_00000003.bin` and `state_00000003_00000.bin`; retained `main.log` and `DONE_00000003`.

### `direct_train_sm120_recovered_x10_20260522`

- Trainer output: `log124M/5090_S`
- Command: `./train-sm120.sh`
- Rationale: rerun the exact user-facing x10 script after the selected stack returned to the fast runtime band in the telemetry probe.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1` on `NVIDIA GeForce RTX 5090`.
- Step timings: 2483.67, 2480.95, 2484.34, 2486.73, 2488.06, 2489.64, 2494.90, 2492.88, 2495.52, 2495.12 ms.
- Average: 2489.793619 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Decision: recovered-band control. The exact script no longer reproduces the `2541.633 ms` slow band; it is `51.839 ms` faster than that degraded direct-script control and `3.339 ms` faster than stable x10. It is still `0.731 ms` slower than promoted direct proof and `19.951 ms` slower than the `new-goal.md` first-three target, so do not update the goal as complete.
- Cleanup: removed `model_00000010.bin` and `state_00000010_00000.bin`; retained `main.log` and `DONE_00000010`.

### `promoted_precompute_grad_scale_recovered_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_precompute_grad_scale_recovered_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_precompute_grad_scale_recovered_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_precompute_grad_scale_recovered_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW" scripts/run_sm120_optimization_round.sh`.
- Rationale: retest the remaining device-side AdamW grad-scale route in the recovered fast band, because the earlier precompute trial happened in a degraded runtime band and this path removes the timed host grad-norm read before AdamW.
- Build/runtime settings: promoted SM120 default stack plus `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW`; CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = precomputed device AdamW scalar`, and `gelu_fusion = 1`.
- Step timings: 2482.35, 2477.72, 2482.46 ms.
- Average: 2480.089545 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609930`.
- Decision: rejected-near-current. This is `0.696 ms` slower than the recovered selected-stack x3, `10.247 ms` slower than the `new-goal.md` target, and only a short-run recovered-band result. It is faster than promoted direct proof by `8.973 ms`, stable x10 by `13.043 ms`, and the recovered exact x10 by `9.704 ms`, but it does not improve over the same-band selected-stack control and does not meet the target.
- Restore verification: rebuilt the promoted default stack without `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_precompute_recovered_20260522`; all nine focused CUDA smokes passed. Restored `train_gpt2cu` sha256 is `4f4ea367835deb19e415eb36db246873c392091edc73dd2050717de047148f61`.
- Cleanup: harness removed candidate checkpoint files; output dir retains `DONE_00000003` and `main.log`.

### `zero_stage0_probe_x3_20260522`

- Trainer output: `log124M/5090_S_zero0_probe_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_zero0_probe_x3_20260522 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 0 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`.
- Rationale: test whether the single-process trainer is faster without the `-z 1` ZeRO-1 code path used by `train-sm120.sh`.
- Runtime settings: current restored selected binary, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `zero_stage=0`, CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2484.12, 2478.42, 2481.70 ms.
- Average: 2480.058670 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609930`.
- Decision: rejected-near-current. This is `0.665 ms` slower than the recovered selected-stack x3 with `-z 1` and `10.216 ms` slower than the `new-goal.md` target, so keep `train-sm120.sh` on `-z 1`.
- Cleanup: removed `model_00000003.bin` and `state_00000003_00000.bin`; retained `main.log` and `DONE_00000003`.

### `precompute_grad_scale_zero0_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_precompute_grad_scale_zero0_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_precompute_grad_scale_zero0_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_precompute_grad_scale_zero0_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=0 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW" scripts/run_sm120_optimization_round.sh`.
- Harness change: `scripts/run_sm120_optimization_round.sh` now exposes `TRAIN_ZERO_STAGE` with default `1`, and `dev/write_sm120_round_manifest.py` records `train_zero_stage`, so single-GPU `-z` variants are tracked instead of being manual-only runs.
- Rationale: combine the two recovered-band near misses, precomputed device AdamW grad scale and no-ZeRO single-GPU training, to test whether their small individual overheads cancel when composed.
- Build/runtime settings: promoted SM120 default stack plus `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW`, `TRAIN_ZERO_STAGE=0`, CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = precomputed device AdamW scalar`, `gelu_fusion = 1`, and `zero_stage = 0`.
- Step timings: 2501.11, 2488.78, 2488.01 ms.
- Average: 2488.395572 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609930`.
- Binary hash: candidate `train_gpt2cu` sha256 `04c4035340ea643f2afc3995dc17d01f9dc8015d441fdaaef1f1d162e4edd5e4`; restored promoted-default `train_gpt2cu` sha256 `a55f2b0ed7bdc347fdd3f2bba301b9ebeb0b53587a323a2a3275b29d3b445303`.
- Restore verification: rebuilt the promoted default stack without `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` and with `TRAIN_ZERO_STAGE=1` in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_precompute_zero0_20260522`; all nine focused CUDA smokes passed.
- Cleanup: harness removed candidate checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: rejected. This is `9.002 ms` slower than the recovered selected-stack x3, `8.306 ms` slower than precomputed grad scale alone, `8.337 ms` slower than `-z 0` alone, and `18.553 ms` slower than the `new-goal.md` target. Keep host-scalar AdamW grad scale and `-z 1`.

### `promoted_backward_n96_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_backward_n96_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_backward_n96_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_backward_n96_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_BACKWARD_N96=1" scripts/run_sm120_optimization_round.sh`.
- Rationale: test the unclosed SM120 TK backward N96 tile selector for GPT-2 hidden/projection shapes divisible by 96.
- Build/runtime settings: promoted SM120 default stack plus `LLMK_SM120_BACKWARD_N96=1`; CUDA-kernel grad-zero route, Torch C++ dresidual-zero route, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark evidence: qkv `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `1012.68/1110.45/1109.38 us` versus `1010.52/994.39/996.36 us`; attproj `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `367.57/371.73/377.85 us` versus `367.32/326.87/330.75 us`; fc `dInp/dW/dW+accum` cuBLASLt/cuBLAS timings `1395.84/1465.49/1494.94 us` versus `1346.43/1306.51/1350.06 us`; fcproj `dInp+dGeLU` TK/cuBLASLt-fused timings `1937.42/1846.78 us`; attention forward/backward `781.213/2742.210 us`; LayerNorm C=768 forward/fused/backward `143.565/282.312/275.432 us`; CUDA grad memset row `158.470 us`; AdamW update row `1812.413 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2491.60, 2486.42, 2488.24 ms.
- Average: 2487.329006 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `2319acc6c1190604bf753404dc2ea9b47b93c2b48cc8058132d438fd5c6108ab`; restored promoted-default `train_gpt2cu` sha256 `79bc5e2e135e7b9710e7140a98aee4b9640990c29f457031f5c6ff0323dfa6d3`.
- Restore verification: rebuilt the promoted default stack without `LLMK_SM120_BACKWARD_N96=1` in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_backward_n96_20260522`; all nine focused CUDA smokes passed.
- Cleanup: harness removed candidate checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: rejected. The N96 tile selector is `7.935 ms` slower than the recovered selected-stack x3, `17.486 ms` slower than the `new-goal.md` target, and only `1.733 ms` faster than promoted direct x10. Do not x10-gate or promote; keep `LLMK_SM120_BACKWARD_N96=0`.

### `promoted_no_cublaslt_gemm_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_no_cublaslt_gemm_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_no_cublaslt_gemm_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_CUBLASLT_GEMM=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Rationale: test a major trainer stack switch by disabling the promoted cuBLASLt GEMM route while keeping the selected CUDA-kernel grad-zero, Torch C++ dresidual-zero, dprep3, block1024, LayerNorm-bwd1, and maxconn1 stack.
- Build/runtime settings: trainer and smoke binaries were built without `LLMK_SM120_USE_CUBLASLT_GEMM`; `bench_sm120_matmul` still carries its explicit cuBLASLt benchmark define, so no matmul row was used as promotion evidence for this candidate.
- Focused correctness: rejected before training. `test_matmul` passed 9/10 checks but failed `dInp backward fused dGELU (GPT-2 fcproj route)` with max abs diff `0.5000` at tolerance `0.50` on RTX 5090. The promoted default stack passed the same route after restoration with max abs diff `0.1250`.
- Binary hash: failed candidate `train_gpt2cu` sha256 `816602d671bb0c73f0438c425e1976cd9b9b8436f5cb70d8ed53fff3be0d7313`; restored promoted-default `train_gpt2cu` sha256 `0817d8477161c7f1b829e1b42563f7038540b39844802f88f6570c7c6488b1fa`.
- Restore verification: rebuilt the promoted default stack with cuBLASLt GEMM enabled in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_no_cublaslt_gemm_20260522`; all nine focused CUDA smokes passed.
- Decision: correctness-rejected. Do not train, x10-gate, or promote a stack that disables cuBLASLt GEMM wholesale until the fcproj fused dGELU route is fixed. Keep the cuBLASLt-backed promoted trainer route selected.

### `promoted_libtorch_grad_zero_recovered_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_libtorch_grad_zero_recovered_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_libtorch_grad_zero_recovered_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_libtorch_grad_zero_recovered_x3_20260522 MAX_STEPS=3 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=1 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Rationale: retest the trainer-callable Torch C++ grad-zero route in the recovered runtime band instead of relying only on older slow-band evidence, while keeping the rest of the promoted stack unchanged.
- Build/runtime settings: promoted SM120 default stack, cuBLASLt GEMM enabled, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, Torch C++ grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, and `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2580.56, 2575.31, 2490.49 ms.
- Average: 2532.897115 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: candidate `train_gpt2cu` sha256 `c1ec88ad1677bf9eed9b6578b4d35255e2463fb96aad191ee1d7b4e75f9434e1`; restored promoted-default `train_gpt2cu` sha256 `4a383d26106f39decf9952eb1283b2bf7e22f5088bcbcdf63cc36cfee7b35c81`.
- Restore verification: rebuilt the promoted default stack with CUDA-kernel grad-zero in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_libtorch_grad_zero_recovered_20260522`; all nine focused CUDA smokes passed.
- Cleanup: harness removed candidate checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: rejected. The Torch C++ grad-zero route is `53.503 ms` slower than the recovered selected-stack x3, `63.054 ms` slower than the `new-goal.md` target, `43.835 ms` slower than promoted direct x10, and `39.764 ms` slower than stable x10. Keep CUDA-kernel grad-zero selected; only the Torch C++ dresidual-zero route remains promoted.

### `direct_train_sm120_post_libtorch_restore_x10_20260522`

- Trainer output: `log124M/5090_S`
- Command: `./train-sm120.sh`
- Rationale: verify the exact user-facing script after rejecting the recovered-band LibTorch grad-zero retest and restoring the promoted CUDA-kernel grad-zero default.
- Runtime settings: active restored promoted default, `CUDA_DEVICE_MAX_CONNECTIONS=1` from the script, cuBLASLt GEMM, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2487.23, 2484.95, 2489.32, 2493.60, 2494.94, 2495.30, 2501.25, 2501.01, 2510.51, 2499.03 ms.
- Average: 2496.657451 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: active restored promoted-default `train_gpt2cu` sha256 `4a383d26106f39decf9952eb1283b2bf7e22f5088bcbcdf63cc36cfee7b35c81`.
- Cleanup: removed `model_00000010.bin` and `state_00000010_00000.bin`; retained `main.log` and `DONE_00000010`.
- Decision: runtime-drift control. The exact script remains in the selected-stack band and is much faster than the severe degraded run, but it is `6.864 ms` slower than the earlier recovered exact x10, `7.595 ms` slower than promoted direct proof, `3.524 ms` slower than stable x10, and `26.814 ms` slower than the `new-goal.md` first-three target. Do not promote a new stack from this run.

### `direct_train_sm120_fresh_slow_control_x10_20260522`

- Artifact file: `scratch/sm120_rounds/codex_sm120_direct_train_sm120_fresh_slow_control_x10_20260522/train-sm120.log`
- Trainer output: `log124M/5090_S`
- Command: `./train-sm120.sh`
- Rationale: answer the fresh slowdown question by rerunning the exact user-facing script without rebuilding or changing candidate flags, then comparing the active backend mix and binary hash to the selected promoted default.
- Runtime settings: active restored promoted default, `CUDA_DEVICE_MAX_CONNECTIONS=1` from the script, cuBLASLt GEMM, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1` on `NVIDIA GeForce RTX 5090`.
- Step timings: 2572.43, 2568.12, 2558.55, 2564.88, 2586.59, 2578.76, 2582.10, 2573.92, 2583.25, 2584.31 ms.
- Average: 2575.607141 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: active restored promoted-default `train_gpt2cu` sha256 `a55f2b0ed7bdc347fdd3f2afc3995dc17d01f9dc8015d441fdaaef1f1d162e4edd5e4`.
- Cleanup: removed `model_00000010.bin` and `state_00000010_00000.bin`; retained `main.log`, copied to the artifact file above, and retained `DONE_00000010`.
- Decision: regression-control. This is the same selected CUDA/Torch stack and restored binary hash, but it is `85.814 ms` slower than the earlier recovered exact x10, `86.545 ms` slower than promoted direct proof, `82.474 ms` slower than stable x10, and `105.764 ms` slower than the `new-goal.md` first-three target. The slowdown is real on the direct script path, but this run does not indicate that a rejected kernel component is active.

### `fresh_rebuild_selected_control_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_fresh_rebuild_selected_control_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_fresh_rebuild_selected_control_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_fresh_rebuild_selected_control_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Rationale: rebuild the selected promoted stack after a severe slow direct-script band to test whether the current source-selected trainer could still recover.
- Build/runtime settings: promoted SM120 default stack with cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2494.83, 2490.90, 2491.66 ms.
- Average: 2491.280079 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: `train_gpt2cu` sha256 `814f797a276d0370208e6a84b33b41f674d139947bbde3cac50f460f887816c9`.
- Cleanup: harness removed checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: control only. The fresh rebuild recovered most of the severe slow band, but it remains `21.437 ms` slower than the `new-goal.md` target and is not goal-complete evidence.

### `precompute_grad_scale_maxconn8_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_precompute_grad_scale_maxconn8_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_precompute_grad_scale_maxconn8_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=8 RUN_LABEL=codex_sm120_precompute_grad_scale_maxconn8_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW" scripts/run_sm120_optimization_round.sh`.
- Rationale: combine the closest recovered-band device-side AdamW grad-scale route with the fastest short-run scheduler signal to test whether removing the host scalar read changes maxconn8 stability.
- Build/runtime settings: promoted SM120 default stack plus `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` and `CUDA_DEVICE_MAX_CONNECTIONS=8`; CUDA-kernel grad-zero, Torch C++ dresidual-zero, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = precomputed device AdamW scalar`, and `gelu_fusion = 1`.
- Step timings: 5470.43, 3290.21, 2465.06 ms.
- Average: 2877.633452 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609930`.
- Binary hash: candidate `train_gpt2cu` sha256 `7ad1d08c7a68f5f5e871b0f9bf4d249d12ffdaeb87d615275abbfb4f60239a30`; selected default restored later to `a9f1277e26ec19923ea838e39dc7c26f7c059710c9cf35f6cc1c613c59c69e8b`.
- Cleanup: harness removed candidate checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: rejected. The composition caused severe warmup and memory-pressure regression, so do not x10-gate or promote.

### `user_observed_train_sm120_fast_first5_20260522`

- Artifact: user-pasted stdout excerpt in the conversation.
- Command: `./train-sm120.sh`.
- Rationale: record the user's faster direct-script observation because it beats the `new-goal.md` first-three target and may identify a runtime state worth reproducing.
- Reported runtime settings: normal selected backend mix with `estimated maximum batch size: 73`, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, and `-z 1`.
- Reported step timings: 2464.71, 2456.65, 2462.99, 2461.49, 2465.34 ms.
- First-three average: 2461.450000 ms.
- First-five average: 2462.236000 ms.
- Decision: promising but unverified. The reported first-three average beats `new-goal.md` by `8.393 ms`, but Codex reruns did not reproduce it, so the goal remains active until the band is captured and reproduced.

### `direct_train_sm120_a9f_verify2_x10_20260522`

- Artifact file: `scratch/sm120_rounds/direct_train_sm120_a9f_verify2_x10_20260522.log`
- Trainer output: `log124M/5090_S`
- Command: `./train-sm120.sh`.
- Rationale: attempt to reproduce the user's faster-than-target direct-script band on the same active selected-backend binary after confirming the GPU was idle.
- Runtime settings: selected promoted stack with cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2508.04, 2498.46, 2505.43, 2505.11, 2509.90, 2510.92, 2514.73, 2515.49, 2521.68, 2519.85 ms.
- Average: 2511.285146 ms.
- First-three average: 2503.976667 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: `train_gpt2cu` sha256 `a9f1277e26ec19923ea838e39dc7c26f7c059710c9cf35f6cc1c613c59c69e8b`.
- Cleanup: removed `model_00000010.bin` and `state_00000010_00000.bin`; retained `main.log` and `DONE_00000010`.
- Decision: reproduction miss. The same selected backend names and active binary did not reproduce the user's 2460 ms band; keep investigating runtime-state conditions before claiming goal completion.

### `direct_train_sm120_telemetry_verify3_x10_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_direct_train_telemetry_verify3_20260522`
- Trainer log: `scratch/sm120_rounds/codex_sm120_direct_train_telemetry_verify3_20260522/train-sm120.log`
- Telemetry log: `scratch/sm120_rounds/codex_sm120_direct_train_telemetry_verify3_20260522/dmon.log`
- Trainer output: `log124M/5090_S`
- Command: `./train-sm120.sh` with background `nvidia-smi dmon` capture.
- Rationale: rerun the exact user-facing script with telemetry after the user's faster first-five observation to separate kernel-stack selection from runtime-state drift.
- Runtime settings: selected promoted stack with cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2482.32, 2478.04, 2479.10, 2481.35, 2483.35, 2487.99, 2487.48, 2490.96, 2491.55, 2492.77 ms.
- Average: 2485.843844 ms.
- First-three average: 2479.820000 ms.
- First-five average: 2480.832000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Telemetry: under sustained load, SM was mostly 99-100%, power about 574-580 W, memory clock 13801 MHz, processor clock mostly 2677-2707 MHz, framebuffer memory about 30353 MiB, and `pviol=0` / `tviol=0`.
- Binary hash: `train_gpt2cu` sha256 `a9f1277e26ec19923ea838e39dc7c26f7c059710c9cf35f6cc1c613c59c69e8b`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `main.log` and `DONE_00000010`.
- Decision: recovered-band reproduction miss. Runtime improved materially versus the prior x10 rerun, but the first-three average is still `9.977 ms` slower than `new-goal.md` and `18.370 ms` slower than the user-pasted first-three band.

### `direct_train_sm120_a9f_warm_verify4_x10_20260522`

- Artifact file: `scratch/sm120_rounds/direct_train_sm120_a9f_warm_verify4_x10_20260522.log`
- Trainer output: `log124M/5090_S`
- Command: `./train-sm120.sh`.
- Rationale: run a warm follow-up immediately after the telemetry reproduction attempt to see whether the recovered runtime band persists without telemetry overhead.
- Runtime settings: selected promoted stack with cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2486.50, 2480.59, 2483.14, 2488.17, 2487.11, 2488.52, 2489.73, 2491.38, 2497.58, 2499.90 ms.
- Average: 2489.568313 ms.
- First-three average: 2483.410000 ms.
- First-five average: 2485.102000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: `train_gpt2cu` sha256 `a9f1277e26ec19923ea838e39dc7c26f7c059710c9cf35f6cc1c613c59c69e8b`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `main.log` and `DONE_00000010`.
- Decision: recovered-band reproduction miss. The warm follow-up remained faster than the slow verify2 run, but it did not reproduce the user's fast band and does not change the selected kernel stack.

### `direct_train_sm120_a9f_verify5_x10_20260522`

- Artifact file: `scratch/sm120_rounds/direct_train_sm120_a9f_verify5_x10_20260522/train-sm120.log`
- Trainer output: `log124M/5090_S`
- Command: `./train-sm120.sh`.
- Rationale: rerun the exact user-facing script on an idle GPU after tracker reconciliation to check whether the user-observed fast band would reproduce.
- Pre-run GPU state: `nvidia-smi` showed no running processes, P8, 46 W, 2611 MiB resident memory, and 1% GPU utilization.
- Runtime settings: selected promoted stack with cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2480.70, 2475.72, 2478.01, 2479.76, 2482.26, 2485.00, 2486.35, 2488.09, 2490.83, 2492.75 ms.
- Average: 2484.308799 ms.
- First-three average: 2478.143333 ms.
- First-five average: 2479.290000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: `train_gpt2cu` sha256 `a9f1277e26ec19923ea838e39dc7c26f7c059710c9cf35f6cc1c613c59c69e8b`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `main.log` and `DONE_00000010`.
- Decision: recovered-band reproduction miss. This is the best Codex reproduction in the current sequence and faster than the promoted x10 proof, but it is still `8.300 ms` slower than `new-goal.md` on first-three average and `16.693 ms` slower than the user-pasted first-three band.

### `direct_train_sm120_a9f_maxconn8_verify6_x10_20260522`

- Artifact file: `scratch/sm120_rounds/direct_train_sm120_a9f_maxconn8_verify6_x10_20260522/train-sm120.log`
- Trainer output: `log124M/5090_S`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=8 ./train-sm120.sh`.
- Rationale: retest the prior short-run scheduler signal in the recovered runtime band without changing the compiled kernel stack.
- Pre-run GPU state: `nvidia-smi` showed no running processes, P8, 46 W, 2584 MiB resident memory, and 0% GPU utilization.
- Runtime settings: selected promoted stack with cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=8`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2482.75, 2476.39, 2479.62, 2481.23, 2482.27, 2486.31, 2488.40, 2489.40, 2491.91, 2495.42 ms.
- Average: 2485.661533 ms.
- First-three average: 2479.586667 ms.
- First-five average: 2480.452000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: `train_gpt2cu` sha256 `a9f1277e26ec19923ea838e39dc7c26f7c059710c9cf35f6cc1c613c59c69e8b`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `main.log` and `DONE_00000010`.
- Decision: rejected recovered-band runtime retest. Maxconn8 is `1.353 ms` slower than the same-band maxconn1 direct rerun on x10 average and `1.443 ms` slower on first-three average, so keep `train-sm120.sh` on `CUDA_DEVICE_MAX_CONNECTIONS=1`.

### `direct_train_sm120_a9f_maxconn16_verify7_x10_20260522`

- Artifact file: `scratch/sm120_rounds/direct_train_sm120_a9f_maxconn16_verify7_x10_20260522/train-sm120.log`
- Trainer output: `log124M/5090_S`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=16 ./train-sm120.sh`.
- Rationale: retest the second prior short-run scheduler signal in the recovered runtime band without changing the compiled kernel stack.
- Pre-run GPU state: `nvidia-smi` showed no running processes, P8, 46 W, 2584 MiB resident memory, and 1% GPU utilization.
- Runtime settings: selected promoted stack with cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=16`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2481.95, 2475.58, 2479.20, 2481.11, 2483.89, 2486.59, 2486.50, 2489.60, 2492.66, 2494.53 ms.
- Average: 2485.516787 ms.
- First-three average: 2478.910000 ms.
- First-five average: 2480.346000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: `train_gpt2cu` sha256 `a9f1277e26ec19923ea838e39dc7c26f7c059710c9cf35f6cc1c613c59c69e8b`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `main.log` and `DONE_00000010`.
- Decision: rejected recovered-band runtime retest. Maxconn16 is `1.208 ms` slower than the same-band maxconn1 direct rerun on x10 average and `0.767 ms` slower on first-three average, so keep `train-sm120.sh` on `CUDA_DEVICE_MAX_CONNECTIONS=1`.

### `runtime_grad_zero_recovered_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_recovered_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_runtime_grad_zero_recovered_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_runtime_grad_zero_recovered_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Rationale: retest the benchmark-vs-trainer mismatch where runtime benchmark rows show `cudaMemsetAsync` competitive for gradient zeroing while the promoted stack uses the CUDA zero kernel.
- Build/runtime settings: promoted SM120 fast default with cuBLASLt GEMM, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and CUDA runtime grad-zero instead of `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2483.42, 2476.45, 2479.69 ms.
- Trainer average: 2478.069782 ms.
- Visible first-three average: 2479.853333 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Candidate binary hash: `train_gpt2cu` sha256 `37d4f87a862312bdb5cd3d86e464a506219a088f89fc3bae3f4ddd239c59ace8`.
- Cleanup: harness removed generated checkpoint files; output dir retains `main.log` and `DONE_00000003`.
- Restore: rebuilt the selected CUDA-kernel grad-zero stack after rejection; active `train_gpt2cu` sha256 is `9fe90db106b68e13757912e30618318e8e6b6686c237108ad2f7dc101db71cd9`.
- Decision: rejected. The trainer-reported average is close because it excludes step 1, but the visible first-three average is `1.710 ms` slower than the selected same-band maxconn1 direct run and `10.010 ms` slower than `new-goal.md`, so do not x10-gate or promote.

### `direct_train_sm120_9fe_restore_verify8_x10_20260522`

- Artifact file: `scratch/sm120_rounds/direct_train_sm120_9fe_restore_verify8_x10_20260522/train-sm120.log`
- Trainer output: `log124M/5090_S`
- Command: `./train-sm120.sh`.
- Rationale: rebaseline the exact direct user-facing path after restoring the selected CUDA-kernel grad-zero build from the rejected CUDA-runtime grad-zero candidate.
- Pre-run GPU state: `nvidia-smi` showed no running processes, P8, 45 W, 2585 MiB resident memory, and 1% GPU utilization.
- Runtime settings: selected promoted stack with cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2481.66, 2475.41, 2477.37, 2482.85, 2483.78, 2487.24, 2490.02, 2492.19, 2489.16, 2493.75 ms.
- Average: 2485.751258 ms.
- First-three average: 2478.146667 ms.
- First-five average: 2480.214000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: `train_gpt2cu` sha256 `9fe90db106b68e13757912e30618318e8e6b6686c237108ad2f7dc101db71cd9`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `main.log` and `DONE_00000010`.
- Decision: recovered-band control. The restored selected stack ties the prior best reproduced first-three within `0.004 ms`, but it remains `8.304 ms` slower than `new-goal.md`, so it is not completion evidence.

### `direct_train_sm120_9fe_user_rerun_verify9_x10_20260522`

- Artifact file: `scratch/sm120_rounds/direct_train_sm120_9fe_user_rerun_verify9_x10_20260522/train-sm120.log`
- Trainer output: `log124M/5090_S`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh`.
- Rationale: user-requested direct rerun after a pasted faster run, with an idle pre-run GPU, to check whether the fast band reproduces under the active selected stack.
- Pre-run GPU state: `nvidia-smi` showed no running processes, P8, 46 W, 2593 MiB resident memory, and 0% GPU utilization.
- Runtime settings: selected promoted stack with cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2482.59, 2477.08, 2478.49, 2481.99, 2483.52, 2485.55, 2487.62, 2490.23, 2491.05, 2493.71 ms.
- Average: 2485.469156 ms.
- First-three average: 2479.386667 ms.
- First-five average: 2480.734000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: `train_gpt2cu` sha256 `9fe90db106b68e13757912e30618318e8e6b6686c237108ad2f7dc101db71cd9`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `main.log` and `DONE_00000010`.
- Decision: recovered-band reproduction miss. This direct rerun is `0.282 ms` faster than the restored selected-stack control on trainer x10 average but `1.240 ms` slower on first-three average; it remains `9.544 ms` slower than `new-goal.md` first-three and `17.937 ms` slower than the user-pasted first-three band, so it is not completion evidence.

### `direct_train_sm120_aa2_user_rerun_verify10_x10_20260522`

- Artifact file: `scratch/sm120_rounds/direct_train_sm120_aa2_user_rerun_verify10_x10_20260522/train-sm120.log`
- Trainer output: `log124M/5090_S`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh`.
- Rationale: user-suggested direct rerun after reporting that the GPU may have been busy during the previous Codex run, to test whether the pasted fast band reproduces on an idle GPU.
- Pre-run GPU state: `nvidia-smi` showed no running processes, P8, 46 W, 2611 MiB resident memory, and 1% GPU utilization.
- Runtime settings: selected promoted stack with cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2483.82, 2477.82, 2481.93, 2485.33, 2496.30, 2490.29, 2491.13, 2494.33, 2497.68, 2498.43 ms.
- Average: 2490.358909 ms.
- First-three average: 2481.190000 ms.
- First-five average: 2485.040000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: `train_gpt2cu` sha256 `aa2d2499d62ab0d4f9fc470fc847c7037a19055b384e5767f31d045aa2ea1bb0`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `main.log` and `DONE_00000010`.
- Decision: recovered-band reproduction miss. This run is `4.890 ms` slower than the previous selected-stack user rerun on trainer x10 average and `1.803 ms` slower on first-three average; it remains `11.347 ms` slower than `new-goal.md` first-three and `19.740 ms` slower than the user-pasted first-three band, so it is not completion evidence.

### `direct_train_sm120_0452_user_followup_verify11_x10_20260522`

- Artifact file: `scratch/sm120_rounds/direct_train_sm120_0452_user_followup_verify11_x10_20260522/train-sm120.log`
- Trainer output: `log124M/5090_S`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh`.
- Rationale: user suggested rerunning after a pasted faster result and after noting the GPU may have been busy in a prior Codex run.
- Pre-run GPU state: `nvidia-smi` showed no running processes, P8, 47 W, 2639 MiB resident memory, and 1% GPU utilization.
- Runtime settings: selected promoted stack with cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2489.49, 2477.76, 2481.13, 2485.99, 2485.96, 2489.23, 2492.06, 2494.38, 2498.67, 2499.11 ms.
- Average: 2489.367194 ms.
- First-three average: 2482.793333 ms.
- First-five average: 2484.066000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Binary hash: `train_gpt2cu` sha256 `0452da6344d3144b24d4c213ddcacf34dbc935f90e04a5288193dc8c42f7f15e`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `main.log` and `DONE_00000010`.
- Decision: recovered-band reproduction miss. This run is `0.992 ms` faster than the previous selected-stack user rerun on trainer x10 average but `1.603 ms` slower on first-three average; it remains `12.950 ms` slower than `new-goal.md` first-three and `21.343 ms` slower than the user-pasted first-three band, so it is not completion evidence.

### `direct_train_sm120_0452_exact_newgoal_x3_20260522`

- Artifact file: `scratch/sm120_rounds/direct_train_sm120_0452_exact_newgoal_x3_20260522/train-sm120-x3.log`
- Trainer output: `log124M/5090_S`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`.
- Rationale: rerun the exact three-step command recorded in `new-goal.md` because the current direct helper script uses `-x 10`.
- Pre-run GPU state: `nvidia-smi` showed no running processes, P8, 48 W, 2640 MiB resident memory, and 1% GPU utilization.
- Runtime settings: selected promoted stack with cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2486.14, 2479.69, 2483.73 ms.
- Trainer average: 2481.709599 ms.
- Visible first-three average: 2483.186667 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Binary hash: `train_gpt2cu` sha256 `0452da6344d3144b24d4c213ddcacf34dbc935f90e04a5288193dc8c42f7f15e`.
- Cleanup: removed generated `model_00000003.bin` and `state_00000003_00000.bin`; retained `main.log` and `DONE_00000003`.
- Decision: exact-command control. The `-x 3` command itself does not explain the gap: visible first3 is `13.344 ms` slower than `new-goal.md` and `21.737 ms` slower than the user-pasted first-three band, so this is not completion evidence.

### `promoted_disable_cuda_profiler_recovered_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_disable_cuda_profiler_recovered_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_disable_cuda_profiler_recovered_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_disable_cuda_profiler_recovered_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_DISABLE_CUDA_PROFILER" scripts/run_sm120_optimization_round.sh`.
- Rationale: retest the profiler-disable source knob on the final selected CUDA-kernel grad-zero, Torch C++ dresidual-zero, dprep3, block1024, LayerNorm-bwd1, maxconn1 stack; older profiler-disable evidence came from pre-final stack contexts.
- Pre-run GPU state: `nvidia-smi` in the round summary showed no running processes, P8, 47 W, 2641 MiB resident memory, and 1% GPU utilization.
- Build/runtime settings: selected promoted stack plus `LLMK_DISABLE_CUDA_PROFILER`; cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2486.78, 2481.02, 2484.63 ms.
- Trainer average: 2482.822776 ms.
- Visible first-three average: 2484.143333 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Candidate binary hash: `train_gpt2cu` sha256 `105e95d7e7317561ea6b3e7e310d0c65200b8603a689367ad2e6334ea64e30fd`.
- Cleanup: harness removed generated checkpoint files; output dir retains `main.log` and `DONE_00000003`.
- Restore: rebuilt the selected stack without `LLMK_DISABLE_CUDA_PROFILER`; active restored `train_gpt2cu` sha256 is `407bd0a44f4515dbdcce54113ee74c802d8c046d3f0681cabd7e8aa0d96dde75`, and all nine focused smokes passed after restoration.
- Decision: rejected. Disabling profiler calls is `1.113 ms` slower than the exact selected-stack x3 control on trainer average, `0.957 ms` slower on visible first3, `14.300 ms` slower than `new-goal.md` first-three, and `22.693 ms` slower than the user-pasted first-three band.

### `dprep2_recovered_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_dprep2_recovered_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_dprep2_recovered_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_dprep2_recovered_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=0 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DPREP_WARPS=2 -DLLMK_SM120_MEMORY_BLOCK_SIZE=1024 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Rationale: test the untried lower attention dprep warp count while preserving the selected trainer stack. `SM120_FAST_TRAINER=0` was required so the Makefile would not override the explicit dprep value with `LLMK_SM120_DPREP_WARPS=3`.
- Pre-run GPU state: `nvidia-smi` in the round summary showed no running processes, P8, 46 W, 2594 MiB resident memory, and 1% GPU utilization.
- Build/runtime settings: cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=2`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Focused attention benchmark: forward `787.510 us`, backward `2737.404 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2490.08, 2483.28, 2486.90 ms.
- Trainer average: 2485.088706 ms.
- Visible first-three average: 2486.753333 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Candidate binary hash: `train_gpt2cu` sha256 `30e70cabfa26dff60c30d15b6d41519ccbc4fed6f62f5e4b2a7a58c13015d5a5`.
- Cleanup: harness removed generated checkpoint files; output dir retains `main.log` and `DONE_00000003`.
- Restore: rebuilt the selected dprep=3 stack with CUDA-kernel grad-zero and Torch C++ dresidual-zero; restored `train_gpt2cu` sha256 is `7d03f3024e013529273883593dfad1ec99b23ccc12475ff406c9f49caacf58f7`, and all nine focused smokes passed after restoration.
- Decision: rejected. The focused attention benchmark did not translate to trainer speed: visible first3 is `7.367 ms` slower than the latest selected-stack direct rerun, `8.607 ms` slower than the restored selected-stack control, and `16.910 ms` slower than `new-goal.md`.

### `dprep1_recovered_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_dprep1_recovered_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_dprep1_recovered_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_dprep1_recovered_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=0 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DPREP_WARPS=1 -DLLMK_SM120_MEMORY_BLOCK_SIZE=1024 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Rationale: close the adjacent lower dprep warp-count sweep after dprep=2 missed, again preserving the selected trainer stack except for `LLMK_SM120_DPREP_WARPS`.
- Pre-run GPU state: `nvidia-smi` showed no running processes, P8, 46 W, 2597 MiB resident memory, and 1% GPU utilization.
- Build/runtime settings: cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=1`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Focused attention benchmark: forward `784.284 us`, backward `2857.293 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2502.26, 2493.90, 2496.10 ms.
- Trainer average: 2494.999766 ms.
- Visible first-three average: 2497.420000 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Candidate binary hash: `train_gpt2cu` sha256 `2d7bf7fa9a275f4ebc29608b2861d12fa7b698f65c4a86955baacb1531ce2d2e`.
- Cleanup: harness removed generated checkpoint files; output dir retains `main.log` and `DONE_00000003`.
- Restore: rebuilt the selected dprep=3 stack with CUDA-kernel grad-zero and Torch C++ dresidual-zero; restored `train_gpt2cu` sha256 is `7d03f3024e013529273883593dfad1ec99b23ccc12475ff406c9f49caacf58f7`, and all nine focused smokes passed after restoration.
- Decision: rejected. dprep=1 slightly improved the focused attention forward row versus dprep=2 but worsened attention backward and end-to-end training; visible first3 is `18.033 ms` slower than the latest selected-stack direct rerun, `19.273 ms` slower than the restored selected-stack control, and `27.577 ms` slower than `new-goal.md`.

### `attn_fwd64_recovered_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_attn_fwd64_recovered_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_attn_fwd64_recovered_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_attn_fwd64_recovered_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_ATTN_FWD_BLOCK=64" scripts/run_sm120_optimization_round.sh`.
- Rationale: test the unrecorded high-side trainer-shaped attention forward tile on the selected stack. Prior current-stack attention tile tests covered `LLMK_SM120_ATTN_FWD_BLOCK=16` and `LLMK_SM120_ATTN_BWD_BLOCK=32`, but not forward block 64 with the selected backward tile.
- Pre-run GPU state: `nvidia-smi` in the round summary showed no running processes, P8, 47 W, 2610 MiB resident memory, and 1% GPU utilization.
- Build/runtime settings: cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_ATTN_FWD_BLOCK=64`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Focused attention benchmark: forward `3223.705 us`, backward `2742.139 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2717.10, 2711.08, 2713.86 ms.
- Trainer average: 2712.467909 ms.
- Visible first-three average: 2714.013333 ms.
- Losses: initial val loss `11.033161`, final val loss `10.609926`.
- Candidate binary hash: `train_gpt2cu` sha256 `82f972812bb4eece21a760307fa7031f3f701d5e87ea5a6cbbd9e607fd0dc8fd`.
- Cleanup: harness removed generated checkpoint files; output dir retains `main.log` and `DONE_00000003`.
- Restore: rebuilt the selected attention tile stack with CUDA-kernel grad-zero and Torch C++ dresidual-zero; restored `train_gpt2cu` sha256 is `aa2d2499d62ab0d4f9fc470fc847c7037a19055b384e5767f31d045aa2ea1bb0`, and all nine focused smokes passed after restoration.
- Decision: rejected. The larger forward tile is not viable: attention forward is about `4.1x` slower than the selected ~785 us band, and visible first3 is `234.627 ms` slower than the latest selected-stack direct rerun, `235.867 ms` slower than the restored selected-stack control, and `244.170 ms` slower than `new-goal.md`.

### `attn_bwd64_recovered_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_attn_bwd64_recovered_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_attn_bwd64_recovered_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_attn_bwd64_recovered_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_ATTN_BWD_BLOCK=64" scripts/run_sm120_optimization_round.sh`.
- Rationale: test the unrecorded high-side trainer-shaped attention backward tile on the selected stack. Prior current-stack attention tile tests covered `LLMK_SM120_ATTN_BWD_BLOCK=32` and `LLMK_SM120_ATTN_FWD_BLOCK=64`, but not backward block 64 with the selected forward tile.
- Pre-run GPU state: `nvidia-smi` showed no running processes, P8, 46 W, 2642 MiB resident memory, and 1% GPU utilization.
- Build/runtime settings: cuBLASLt GEMM, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_ATTN_BWD_BLOCK=64`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Focused attention benchmark: forward `787.890 us`, backward `18050.102 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 3932.81, 3901.03, 3902.40 ms.
- Trainer average: 3901.718855 ms.
- Visible first-three average: 3912.080000 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Candidate binary hash: `train_gpt2cu` sha256 `72c545ca1f29ce0ff19e6a5ee15b8317fd25317120de8a18791901d708993cfe`.
- Cleanup: harness removed generated checkpoint files; output dir retains `main.log` and `DONE_00000003`.
- Restore: rebuilt the selected attention tile stack with CUDA-kernel grad-zero and Torch C++ dresidual-zero; restored `train_gpt2cu` sha256 is `407bd0a44f4515dbdcce54113ee74c802d8c046d3f0681cabd7e8aa0d96dde75`, and all nine focused smokes passed after restoration.
- Decision: rejected. The larger backward tile is not viable: attention backward is about `6.6x` slower than the selected ~2717 us band, and visible first3 is `1430.890 ms` slower than the latest selected-stack direct rerun, `1433.933 ms` slower than the restored selected-stack control, and `1442.237 ms` slower than `new-goal.md`.

### `attention_refresh_cudnn_torch_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_attention_refresh_20260522`
- Rationale: refresh the highest-payoff remaining attention lead before writing a trainer integration. The current project notes say a cuDNN or Torch route should only be promoted if a trainer-shaped packed/layout route beats packed TK, not merely because separated Q/K/V reference rows are faster.
- Pre-run GPU state: `nvidia-smi` showed no running processes, P8, 46 W, 2640 MiB resident memory, and 1% GPU utilization.
- Native packed TK benchmark: `LLMK_BENCH_REPEATS=9 ./bench_sm120_attention`.
- Native packed TK results: forward `785.153 us`, backward `2734.630 us`, total `3519.783 us`.
- cuDNN command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_cudnn_attention.py --repeats 9 --warmup 3`.
- cuDNN separated results: forward `676.243 us`, backward `2387.222 us`, total `3063.465 us`; keep as reference/layout-rewrite evidence only.
- cuDNN packed results: forward `804.018 us`, backward `2809.104 us`, total `3613.122 us`; saved-forward backward route was active.
- Torch command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_attention.py --repeats 9 --warmup 3`.
- Torch separated results: forward `557.845 us`, backward `2195.517 us`, total `2753.362 us`; keep as reference/layout-rewrite evidence only.
- Torch packed results: forward `1161.370 us`, backward `4065.203 us`, total `5226.573 us`.
- Torch materialized packed results: forward `1260.134 us`, backward `4198.688 us`, total `5458.822 us`.
- Torch qkv-layout command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_attention_layouts.py --repeats 5 --warmup 2 --json-out scratch/sm120_rounds/codex_sm120_attention_refresh_20260522/bench_sm120_torch_attention_layouts.json`.
- Torch qkv-layout results: single-packed `2050.400/6371.072 us`, split-strided `2110.432/4850.400 us`, split-materialized `2661.728/5379.680 us` for forward/backward.
- Decision: benchmark-rejected for trainer integration. cuDNN packed is `93.339 us` slower than packed TK total, Torch packed is `1706.790 us` slower, and the qkv-layout routes are much slower once projection/layout costs are included. Do not implement a cuDNN or Torch attention trainer route without a new packed/layout result that materially beats TK.

### `runtime_grad_zero_recovered_x10_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_recovered_x10_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_runtime_grad_zero_recovered_x10_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_runtime_grad_zero_recovered_x10_20260522 MAX_STEPS=10 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Rationale: run the x10 stability gate for the recovered-band CUDA runtime grad-zero near-miss before considering any replacement of the selected CUDA-kernel grad-zero route.
- Pre-run GPU state: `nvidia-smi` showed no running processes, P8, 47 W, 2638 MiB resident memory, and 1% GPU utilization.
- Build/runtime settings: cuBLASLt GEMM, CUDA runtime grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2485.28, 2477.70, 2481.06, 2484.37, 2485.04, 2489.55, 2491.51, 2493.13, 2499.15, 2500.79 ms.
- Average: 2489.145782 ms.
- First-three average: 2481.346667 ms.
- First-five average: 2482.690000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Candidate binary hash: `train_gpt2cu` sha256 `de8ef2a1244266dbfe851c21fb21b158b4653487356c142f2f4fd04ea3c3bcc0`.
- Cleanup: harness removed generated checkpoint files; output dir retains `main.log` and `DONE_00000010`.
- Restore: rebuilt the selected CUDA-kernel grad-zero stack with Torch C++ dresidual-zero; active restored `train_gpt2cu` sha256 is `0452da6344d3144b24d4c213ddcacf34dbc935f90e04a5288193dc8c42f7f15e`, and all nine focused smokes passed after restoration.
- Decision: rejected-near-current. The x10 average is `1.213 ms` faster than the latest direct selected-stack rerun and `3.987 ms` faster than stable x10, but it is `0.084 ms` slower than promoted direct proof and the visible first3 is `11.504 ms` slower than `new-goal.md`. Do not replace the selected CUDA-kernel grad-zero route.

### `direct_train_sm120_407b_user_idle_rerun_20260522_0601`

- Artifact directory: `scratch/sm120_rounds/direct_train_sm120_407b_user_idle_rerun_20260522_0601`
- Trainer log: `scratch/sm120_rounds/direct_train_sm120_407b_user_idle_rerun_20260522_0601/train-sm120.log`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh`.
- Rationale: rerun the exact selected `train-sm120.sh` path after the user reported a faster `2461.450 ms` first-three band and noted the prior Codex run might have hit a busy GPU.
- Pre-run GPU state: `nvidia-smi` showed no running processes, P8, 47 W, 2641 MiB resident memory, and 1% GPU utilization.
- Build/runtime settings: cuBLASLt GEMM, current promoted attention route with default tiles, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2486.88, 2479.96, 2486.13, 2486.74, 2491.03, 2492.87, 2494.46, 2499.02, 2500.74, 2501.64 ms.
- Trainer average: 2492.510584 ms.
- Visible first-three average: 2484.323333 ms.
- Visible first-five average: 2486.148000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Active binary hash: `train_gpt2cu` sha256 `407bd0a44f4515dbdcce54113ee74c802d8c046d3f0681cabd7e8aa0d96dde75`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `log124M/5090_S`, `main.log`, and `DONE_00000010`.
- Decision: reproduction miss. Even with an idle pre-run GPU, this rerun is `14.480 ms` slower than the `new-goal.md` first-three target and `22.873 ms` slower than the user's pasted first-three band; it is also `3.143 ms` slower than the prior direct selected-stack x10 average. Keep the selected stack active, but do not treat this as target-beating evidence.

### `promoted_matmul_dbias1024_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_matmul_dbias1024_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_matmul_dbias1024_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS=-DLLMK_SM120_BIAS_BLOCK_SIZE=1024 scripts/run_sm120_optimization_round.sh`.
- Rationale: test the previously unrecorded matmul backward-bias reduction block-size knob on the selected promoted stack. This is distinct from the already-tested bias-add block-size hooks and affects the dbias reductions launched by `matmul_backward`.
- Build/runtime settings: selected promoted stack plus `LLMK_SM120_BIAS_BLOCK_SIZE=1024`; cuBLASLt GEMM, current promoted attention route, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: attention `783.716/2741.899 us` forward/backward; LayerNorm `140.045/275.665/274.941 us` forward/fused-residual/backward; runtime bias-grad reduce `27.122/188.627/249.699 us` for OC `768/2304/3072`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2494.12, 2484.60, 2489.08 ms.
- Trainer average: 2486.838818 ms.
- Visible first-three average: 2489.266667 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609922`.
- Candidate binary hash: `train_gpt2cu` sha256 `d33444c91ece267faf43e36530d8c468bf8ff30ab07724f0e73b5e94a59c7298`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Decision: rejected. The 1024-thread dbias reduction variant is `19.423 ms` slower than `new-goal.md` first-three and slower than the selected-stack x3 controls, so no x10 gate or source promotion is justified.

### `promoted_matmul_dbias256_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias256_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_promoted_matmul_dbias256_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_promoted_matmul_dbias256_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS=-DLLMK_SM120_BIAS_BLOCK_SIZE=256 scripts/run_sm120_optimization_round.sh`.
- Rationale: complete the paired sweep for the previously unrecorded matmul backward-bias reduction block-size knob before restoring the selected default.
- Build/runtime settings: selected promoted stack plus `LLMK_SM120_BIAS_BLOCK_SIZE=256`; cuBLASLt GEMM, current promoted attention route, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: attention `782.812/2743.249 us` forward/backward; LayerNorm `137.140/275.838/272.764 us` forward/fused-residual/backward; runtime bias-grad reduce `25.394/186.931/246.125 us` for OC `768/2304/3072`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2500.47, 2491.61, 2494.21 ms.
- Trainer average: 2492.912173 ms.
- Visible first-three average: 2495.430000 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Candidate binary hash: `train_gpt2cu` sha256 `e8ccfa7fa34cbe175ffab1cd3f6c9525c3fccf4f59c703e3bd74c7cec626b6b5`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack without `LLMK_SM120_BIAS_BLOCK_SIZE`; restored `train_gpt2cu` sha256 is `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The 256-thread dbias reduction variant is `25.587 ms` slower than `new-goal.md` first-three and slower than the 1024-thread variant, so keep the default 512-thread dbias reduction block.

### `direct_train_sm120_user_rerun_20260522_0617`

- Artifact directory: `scratch/sm120_rounds/direct_train_sm120_user_rerun_20260522_0617`
- Trainer log: `scratch/sm120_rounds/direct_train_sm120_user_rerun_20260522_0617/train-sm120.log`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh`.
- Rationale: rerun the exact selected `train-sm120.sh` path after the user reported a faster idle-GPU band of `2464.71`, `2456.65`, `2462.99`, `2461.49`, and `2465.34 ms` for the first five steps.
- Pre-run GPU state: `nvidia-smi` showed no running compute processes, P8, 47.27 W, 2660 MiB resident memory, 1% GPU utilization, 345 MHz SM clock, and 405 MHz memory clock.
- Build/runtime settings: cuBLASLt GEMM, current promoted attention route with default tiles, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`.
- Step timings: 2486.77, 2479.92, 2485.18, 2488.32, 2492.04, 2492.45, 2498.59, 2502.10, 2501.67, 2502.72 ms.
- Trainer average: 2493.667046 ms.
- Visible first-three average: 2483.956667 ms.
- Visible first-five average: 2486.446000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Active binary hash: `train_gpt2cu` sha256 `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `log124M/5090_S`, `main.log`, and `DONE_00000010`.
- Decision: reproduction miss. This run is `14.114 ms` slower than the `new-goal.md` first-three target and `22.507 ms` slower than the user's pasted first-three band; it is also slower than the previous direct selected-stack reruns, so it does not justify a kernel-stack change.

### `fcproj_runtime_grad_zero_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_fcproj_runtime_grad_zero_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_fcproj_runtime_grad_zero_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_fcproj_runtime_grad_zero_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS=-DLLMK_SM120_USE_CUBLAS_DINP_FCPROJ scripts/run_sm120_optimization_round.sh`.
- Rationale: test an unrecorded trainer-callable interaction between the near-miss MLP-projection direct-cuBLAS dInput selector and CUDA runtime grad-zero, while retaining the selected cuBLASLt GEMM, Torch C++ dresidual-zero, dprep3, memory block1024, LayerNorm-bwd1, maxconn1, and ZeRO-1 stack.
- Build/runtime settings: cuBLASLt GEMM plus `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ`, CUDA runtime grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: fcproj dInput `TK 1515.94 us / cuBLASLt 1430.76 us / cuBLAS 1383.77 us`; fcproj dInput+dGeLU `TK 1833.59 us / cuBLASLt fused 1860.65 us`; attention `784.393/2743.847 us` forward/backward; LayerNorm `138.650/276.217/273.353 us` forward/fused-residual/backward; runtime grad memset `CUDA runtime 148.870 us / CUDA kernel 150.982 us`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2494.73, 2488.33, 2491.77 ms.
- Trainer average: 2490.052581 ms.
- Visible first-three average: 2491.610000 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Candidate binary hash: `train_gpt2cu` sha256 `234141d3bdf56d84e89ece5e8ec3decf5a986a3373a6a3b0608782ec2130885d`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack without `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` and with CUDA-kernel grad-zero restored; restored `train_gpt2cu` sha256 is `5f5decae7b99c5875d95eeda427b3930031fe29fb73cf832872c21eff719d1a7`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The interaction is correctness-clean, but first3 is `21.767 ms` slower than `new-goal.md` and `7.653 ms` slower than the latest direct selected-stack rerun. The CUDA runtime grad-zero plus fcproj direct-cuBLAS dInput combination does not recover the user's faster band or justify x10 gating.

### `fcproj_precompute_grad_scale_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_fcproj_precompute_grad_scale_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_fcproj_precompute_grad_scale_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_fcproj_precompute_grad_scale_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=0 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_FCPROJ -DLLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW" scripts/run_sm120_optimization_round.sh`.
- Rationale: test an unrecorded interaction between the near-miss MLP-projection direct-cuBLAS dInput selector and the precomputed device AdamW grad-scale route, while retaining selected CUDA-kernel grad-zero, Torch C++ dresidual-zero, dprep3, memory block1024, LayerNorm-bwd1, maxconn1, and ZeRO-1.
- Build/runtime settings: cuBLASLt GEMM plus `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ`, CUDA-kernel grad-zero, Torch C++ dresidual-zero, precomputed device AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = precomputed device AdamW scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2487.66, 2481.16, 2489.98 ms.
- Trainer average: 2485.572457 ms.
- Visible first-three average: 2486.266667 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609930`.
- Candidate binary hash: `train_gpt2cu` sha256 `b8b27b761a5da7788a3f6b5d28d290aae1027c72d50963f1930e60c0940b9186`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack without `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` or `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW`; restored `train_gpt2cu` sha256 is `067f00c38343776b7cd349d3d5cc9bfd3644146510d84f41db15b863ec781af0`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The interaction is `4.480 ms` faster than the fcproj+runtime-grad-zero candidate, but it is still `16.424 ms` slower than `new-goal.md`, `2.310 ms` slower than the latest direct selected-stack rerun, and `6.877 ms` slower than the recovered selected-stack x3 control. Do not x10-gate or promote.

### `tk_dgelu_runtime_grad_zero_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_tk_dgelu_runtime_grad_zero_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_tk_dgelu_runtime_grad_zero_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_tk_dgelu_runtime_grad_zero_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS=-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP scripts/run_sm120_optimization_round.sh`.
- Rationale: test the remaining trainer-callable interaction between the TK fused dGELU dInput selector and CUDA runtime grad-zero, while retaining the selected cuBLASLt GEMM, Torch C++ dresidual-zero, dprep3, memory block1024, LayerNorm-bwd1, maxconn1, and ZeRO-1 stack.
- Build/runtime settings: cuBLASLt GEMM plus `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`, CUDA runtime grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: fcproj dInput+dGeLU `TK 1781.13 us / cuBLASLt fused 1812.51 us / cuBLASLt explicit 2192.33 us / cuBLAS explicit 2165.57 us`; attention `783.432/2741.513 us` forward/backward; LayerNorm `137.638/275.542/271.323 us` forward/fused-residual/backward; runtime grad memset `CUDA runtime 149.331 us / CUDA kernel 149.835 us`; AdamW update `1811.402 us`; encoder forward `79.697 us`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2497.46, 2493.94, 2499.48 ms.
- Trainer average: 2496.712327 ms.
- Visible first-three average: 2496.960000 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609915`.
- Candidate binary hash: `train_gpt2cu` sha256 `ba9d126b2f4f9c5cd680b72dcf8437b637777fab2f76b99bbdd6f3ac4f5e63ce`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack without `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` and with CUDA-kernel grad-zero restored; restored `train_gpt2cu` sha256 is `47f15e2417e6edf90d5b7e4736ae12633f34786882a686c912407988c00ed8a1`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The TK fused dGELU row wins one focused benchmark, but the end-to-end trainer first3 is `27.117 ms` slower than `new-goal.md`, `13.003 ms` slower than the latest direct selected-stack rerun, and `10.693 ms` slower than fcproj+runtime-grad-zero. Do not x10-gate or promote.

### `direct_train_sm120_after_user_2462_band_20260522_0640`

- Artifact directory: `scratch/sm120_rounds/direct_train_sm120_after_user_2462_band_20260522_0640`
- Trainer log: `scratch/sm120_rounds/direct_train_sm120_after_user_2462_band_20260522_0640/train-sm120.log`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh`.
- Rationale: rerun the exact selected `train-sm120.sh` path after the user reported their first five steps in a faster `2456-2465 ms` band and noted the GPU may have been busy during the earlier reproduction attempt.
- Pre-run GPU state: `nvidia-smi` showed no running compute processes, P8, 47 W, 2677 MiB resident memory, and 0% GPU utilization.
- Build/runtime settings: cuBLASLt GEMM, current promoted attention route with default tiles, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2488.58, 2483.47, 2485.15, 2488.12, 2490.31, 2492.06, 2500.91, 2509.12, 2503.65, 2505.38 ms.
- Trainer average: 2495.353434 ms.
- Visible x10 average: 2494.675000 ms.
- Visible first-three average: 2485.733333 ms.
- Visible first-five average: 2487.126000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Active binary hash: `train_gpt2cu` sha256 `47f15e2417e6edf90d5b7e4736ae12633f34786882a686c912407988c00ed8a1`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `log124M/5090_S`, `main.log`, and `DONE_00000010`.
- Decision: reproduction miss. This run is `15.890 ms` slower than the `new-goal.md` first-three target and `24.283 ms` slower than the user's pasted first-three band, so it does not justify changing the selected kernel stack.

### `disable_cublas_bwd_runtime_grad_zero_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_disable_cublas_bwd_runtime_grad_zero_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_disable_cublas_bwd_runtime_grad_zero_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_disable_cublas_bwd_runtime_grad_zero_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS=-DLLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM scripts/run_sm120_optimization_round.sh`.
- Rationale: test the remaining trainer-callable interaction between `LLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM` and CUDA runtime grad-zero. Disabling the direct-cuBLAS backward selector helped a degraded current band, while runtime grad-zero was near-current in x10, so the combination was worth a full trainer smoke instead of relying on the individual scorecard rows.
- Build/runtime settings: cuBLASLt GEMM, direct-cuBLAS backward selector disabled by `LLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM`, CUDA runtime grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: attention `782.554/2741.460 us` forward/backward; LayerNorm `137.655/272.982/271.012 us` forward/fused-residual/backward for `C=768`; runtime grad memset `CUDA runtime 149.314 us / CUDA kernel 149.952 us`; fcproj dInput+dGeLU `TK 1828.38 us / cuBLASLt fused 1836.28 us`; AdamW update `1833.149 us`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2524.06, 2517.30, 2520.63 ms.
- Trainer average: 2518.966794 ms.
- Visible first-three average: 2520.663333 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609921`.
- Candidate binary hash: `train_gpt2cu` sha256 `32995287c01180f3f53db5b386c464ce43ffe61ea98db1002afc6532c8924867`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack without `LLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM` and with CUDA-kernel grad-zero restored; restored `train_gpt2cu` sha256 is `1cfb77e2f9b59d3b54960b64d6ef23fb49a13ca168a0f7d6ba130af8bb9e0608`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The combination is correctness-clean but is `50.820 ms` slower than `new-goal.md` first-three and `34.930 ms` slower than the latest direct selected-stack rerun; it is only `0.554 ms` faster than disabling cuBLAS backward alone in a degraded band, which is not enough to justify x10 gating or promotion.

### `fcproj_zero0_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_fcproj_zero0_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_fcproj_zero0_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_fcproj_zero0_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=0 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS=-DLLMK_SM120_USE_CUBLAS_DINP_FCPROJ scripts/run_sm120_optimization_round.sh`.
- Rationale: test the remaining trainer-callable interaction between the near-miss MLP-projection direct-cuBLAS dInput selector and single-GPU no-ZeRO training. The individual rows were both close enough to justify a composed trainer smoke, but not enough to promote from scorecard evidence alone.
- Build/runtime settings: cuBLASLt GEMM plus `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ`, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 0`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: fcproj dInput `TK 1524.79 us / cuBLASLt 1379.78 us / cuBLAS 1402.16 us`; fcproj dInput+dGeLU `TK 1785.73 us / cuBLASLt fused 1815.24 us`; attention `783.143/2740.611 us` forward/backward; LayerNorm `140.691/276.145/271.881 us` forward/fused-residual/backward; runtime grad memset `CUDA runtime 151.280 us / CUDA kernel 149.798 us`; AdamW update `1835.830 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 0`.
- Step timings: 2496.64, 2490.95, 2492.34 ms.
- Trainer average: 2491.643667 ms.
- Visible first-three average: 2493.310000 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609930`.
- Candidate binary hash: `train_gpt2cu` sha256 `93d4c978602275b26caee28065a0e199b95a2f700dfef540463776e7100a7e51`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack without `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` and with `TRAIN_ZERO_STAGE=1`; restored `train_gpt2cu` sha256 is `e43a87cf1412a4ef303cb6c6fefadb79373176adc55c241a0f1953123b063c5b`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The composition is `23.467 ms` slower than `new-goal.md`, `7.577 ms` slower than the latest direct selected-stack rerun, `7.043 ms` slower than fcproj+precomputed-grad-scale, `1.700 ms` slower than fcproj+runtime-grad-zero, and `13.251 ms` slower than no-ZeRO alone. Do not x10-gate or promote.

### `tk_dgelu_precompute_grad_scale_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP -DLLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW" scripts/run_sm120_optimization_round.sh`.
- Rationale: test the remaining trainer-callable interaction between the TK fused dGELU dInput selector and the precomputed device AdamW grad-scale route. TK continues to win the focused fcproj dInput+dGeLU benchmark row, and precomputed grad-scale has been a near-current trainer route, so this composition needed an end-to-end trainer smoke instead of relying on the native scorecard.
- Build/runtime settings: cuBLASLt GEMM plus `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`, CUDA-kernel grad-zero, Torch C++ dresidual-zero, precomputed device AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: fcproj dInput+dGeLU `TK 1790.50 us / cuBLASLt fused 1861.00 us / cuBLASLt explicit 2181.84 us / cuBLAS explicit 2179.14 us`; attention `782.792/2739.670 us` forward/backward; LayerNorm `137.249/275.846/270.832 us` forward/fused-residual/backward; runtime grad memset `CUDA runtime 148.923 us / CUDA kernel 149.744 us`; AdamW update `1828.720 us`; encoder forward `79.455 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = precomputed device AdamW scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2497.61, 2497.02, 2500.84 ms.
- Trainer average: 2498.933315 ms.
- Visible first-three average: 2498.490000 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609920`.
- Candidate binary hash: `train_gpt2cu` sha256 `e28bcb3ad4a65f8da49112196f17c7a34c354574825e1884cd424b8923e1396a`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack without `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` or `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW`; restored `train_gpt2cu` sha256 is `e42983fb2e98d1c3c81125beb62465aec1b86220efaf4f8ea876e1bbd6d8f280`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The TK dGELU row again wins the focused native benchmark, but the end-to-end trainer first3 is `28.647 ms` slower than `new-goal.md`, `12.757 ms` slower than the latest direct selected-stack rerun, `1.530 ms` slower than TK dGELU plus runtime grad-zero, and `18.400 ms` slower than precomputed grad-scale alone. Do not x10-gate or promote.

### `fcproj_precompute_zero0_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_fcproj_precompute_zero0_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_fcproj_precompute_zero0_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_fcproj_precompute_zero0_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=0 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_FCPROJ -DLLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW" scripts/run_sm120_optimization_round.sh`.
- Rationale: compose the near-miss fcproj direct-cuBLAS dInput selector, precomputed device AdamW grad scale, and single-GPU no-ZeRO path. The component rows were close enough that the interaction needed a trainer run rather than a scorecard-only decision.
- Build/runtime settings: cuBLASLt GEMM plus `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ`, CUDA-kernel grad-zero, Torch C++ dresidual-zero, precomputed device AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 0`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: fcproj dInput `TK 1529.60 us / cuBLASLt 1384.84 us / cuBLAS 1429.41 us`; fcproj dInput+dGeLU `TK 1819.31 us / cuBLASLt fused 1860.84 us`; attention `782.054/2739.630 us` forward/backward; LayerNorm `139.140/276.036/271.039 us` forward/fused-residual/backward; runtime grad memset `CUDA runtime 149.747 us / CUDA kernel 150.131 us`; AdamW update `1833.978 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = precomputed device AdamW scalar`, `gelu_fusion = 1`, and `zero_stage = 0`.
- Step timings: 2494.59, 2488.71, 2491.80 ms.
- Trainer average: 2490.259051 ms.
- Visible first-three average: 2491.700000 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609930`.
- Candidate binary hash: `train_gpt2cu` sha256 `301760416174a60975ac2f865c12fa5b664196714565b44defcb2ddce7ee44f0`.
- Cleanup: no generated checkpoint files remain in the rejected candidate output directory.
- Restore: rebuilt the selected default stack without `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` or `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW`; restored `train_gpt2cu` sha256 is `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The triple composition is correctness-clean and improves over fcproj+no-ZeRO alone, but it is `21.857 ms` slower than `new-goal.md`, `5.967 ms` slower than the latest direct selected-stack rerun before this entry, and slower than the better fcproj/precompute and precompute+no-ZeRO rows. Do not x10-gate or promote.

### `direct_train_sm120_user_followup_rerun_20260522_0704`

- Artifact directory: `scratch/sm120_rounds/direct_train_sm120_user_followup_rerun_20260522_0704`
- Trainer log: `scratch/sm120_rounds/direct_train_sm120_user_followup_rerun_20260522_0704/train-sm120.log`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh`.
- Rationale: rerun the exact selected `train-sm120.sh` path after the user reported a faster first-five band around `2456-2465 ms` and noted that earlier runs may have been affected by other GPU activity.
- Pre-run GPU state: `nvidia-smi` showed no running compute processes, P8, 46 W, 2680 MiB resident memory, and 1% GPU utilization.
- Build/runtime settings: selected cuBLASLt-backed trainer route, current promoted attention route with default tiles, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2489.27, 2483.65, 2487.86, 2488.85, 2493.01, 2496.94, 2500.06, 2501.60, 2504.85, 2505.55 ms.
- Trainer average: 2495.818721 ms.
- Visible x10 average: 2495.164000 ms.
- Visible first-three average: 2486.926667 ms.
- Visible first-five average: 2488.528000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Active binary hash: `train_gpt2cu` sha256 `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `log124M/5090_S`, `main.log`, and `DONE_00000010`.
- Decision: reproduction miss. This run is `17.083 ms` slower than the `new-goal.md` first-three target and `25.477 ms` slower than the user's pasted first-three band, so it does not justify changing the selected kernel stack.

### `runtime_grad_zero_precompute_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_precompute_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_runtime_grad_zero_precompute_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_runtime_grad_zero_precompute_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS=-DLLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW scripts/run_sm120_optimization_round.sh`.
- Rationale: compose the recovered-band CUDA-runtime grad-zero route with the precomputed device AdamW grad-scale route, without adding the already-regressed fcproj, TK dGELU, or no-ZeRO interactions. Runtime grad-zero had the closest recent x10 average, and precompute was a near-current x3 row, so this was the next trainer-callable interaction to test.
- Build/runtime settings: cuBLASLt GEMM, current promoted attention route with default tiles, CUDA runtime grad-zero, Torch C++ dresidual-zero, precomputed device AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: qkv dInput `TK 1090.25 us / cuBLASLt 1015.73 us / cuBLAS 1017.20 us`; fcproj dInput+dGeLU `TK 1806.86 us / cuBLASLt fused 1817.46 us`; attention `784.577/2736.616 us` forward/backward; LayerNorm `137.957/275.869/272.223 us` forward/fused-residual/backward; runtime grad memset `CUDA runtime 148.696 us / CUDA kernel 149.750 us`; AdamW update `1810.045 us`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = precomputed device AdamW scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2498.84, 2492.03, 2495.86 ms.
- Trainer average: 2493.945599 ms.
- Visible first-three average: 2495.576667 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609930`.
- Candidate binary hash: `train_gpt2cu` sha256 `064e1e95dc83339da662d37fd658aefa7702bc48d1c5d6ab4e88895234a862e0`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack with CUDA-kernel grad-zero and host scalar AdamW grad scale restored; restored `train_gpt2cu` sha256 is `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The composition is correctness-clean, but it is `25.733 ms` slower than `new-goal.md`, `8.650 ms` slower than the latest direct selected-stack rerun, `14.230 ms` slower than runtime-grad-zero x10 first3, and `9.310 ms` slower than fcproj+precompute. Do not x10-gate or promote.

### `runtime_grad_zero_zero0_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_zero0_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_runtime_grad_zero_zero0_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_runtime_grad_zero_zero0_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=0 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 scripts/run_sm120_optimization_round.sh`.
- Rationale: compose the CUDA-runtime grad-zero route with single-GPU no-ZeRO training. Both had near-current individual rows, so the interaction needed a trainer run rather than a scorecard-only decision.
- Build/runtime settings: cuBLASLt GEMM, current promoted attention route with default tiles, CUDA runtime grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 0`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: qkv dInput `TK 1088.46 us / cuBLASLt 1024.13 us / cuBLAS 1011.99 us`; fcproj dInput+dGeLU `TK 1819.11 us / cuBLASLt fused 1855.05 us`; attention `784.327/2741.013 us` forward/backward; LayerNorm `135.677/275.735/271.005 us` forward/fused-residual/backward; runtime grad memset `CUDA runtime 148.910 us / CUDA kernel 149.654 us`; AdamW update `1804.282 us`; encoder forward `79.796 us`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 0`.
- Step timings: 2497.47, 2491.92, 2496.83 ms.
- Trainer average: 2494.379759 ms.
- Visible first-three average: 2495.406667 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609930`.
- Candidate binary hash: `train_gpt2cu` sha256 `20bef188744eb105e71c7ef63b8caa5031a6038394a6e1da66c9dd2914d6a560`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack with CUDA-kernel grad-zero and ZeRO stage 1 restored; restored `train_gpt2cu` sha256 is `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The composition is correctness-clean, but it is `25.563 ms` slower than `new-goal.md`, `8.480 ms` slower than the latest direct selected-stack rerun before this entry, and slower than both better component rows. Do not x10-gate or promote.

### `direct_train_sm120_user_followup_rerun_20260522_0721`

- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh`.
- Rationale: rerun the exact selected `train-sm120.sh` path after the user reported a faster first-five band around `2456-2465 ms` and noted that earlier runs may have been affected by other GPU activity.
- Post-run GPU state: `nvidia-smi` showed no running compute processes, P8, 49 W, 2682 MiB resident memory, and 0% GPU utilization; a follow-up clock query showed P8 idle clocks after training.
- Build/runtime settings: selected cuBLASLt-backed trainer route, current promoted attention route with default tiles, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2491.29, 2486.02, 2487.97, 2491.41, 2494.46, 2499.49, 2501.68, 2503.84, 2506.13, 2507.12 ms.
- Trainer average: 2497.570541 ms.
- Visible x10 average: 2496.941000 ms.
- Visible first-three average: 2488.426667 ms.
- Visible first-five average: 2490.230000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Active binary hash: `train_gpt2cu` sha256 `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `log124M/5090_S`, `main.log`, and `DONE_00000010`.
- Decision: reproduction miss. This run is `18.584 ms` slower than the `new-goal.md` first-three target, `26.977 ms` slower than the user's pasted first-three band, and `27.994 ms` slower than the user's pasted first-five band, so it does not justify changing the selected kernel stack.

### `runtime_grad_zero_dprep2_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_runtime_grad_zero_dprep2_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_runtime_grad_zero_dprep2_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=0 SM120_USE_CUDA_KERNEL_GRAD_ZERO=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_DPREP_WARPS=2 -DLLMK_SM120_MEMORY_BLOCK_SIZE=1024 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" scripts/run_sm120_optimization_round.sh`.
- Rationale: test the remaining plausible cheap interaction between the close CUDA-runtime grad-zero row and the dprep=2 attention-prep setting, without adding already-regressed fcproj, TK dGELU, precomputed-grad-scale, or no-ZeRO changes.
- Build/runtime settings: cuBLASLt GEMM, current promoted attention route with `LLMK_SM120_DPREP_WARPS=2`, CUDA runtime grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: qkv dInput `TK 1090.44 us / cuBLASLt 1017.45 us / cuBLAS 1012.20 us`; fcproj dInput+dGeLU `TK 1826.46 us / cuBLASLt fused 1839.35 us`; attention `783.837/2744.174 us` forward/backward; LayerNorm `136.619/275.750/272.306 us` forward/fused-residual/backward; runtime grad memset `CUDA runtime 148.794 us / CUDA kernel 150.083 us`; AdamW update `1805.322 us`; encoder forward `73.967 us`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2500.46, 2494.79, 2499.50 ms.
- Trainer average: 2497.146368 ms.
- Visible first-three average: 2498.250000 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Candidate binary hash: `train_gpt2cu` sha256 `6e62499a9314f357cb23646de4cfb5330bfc4b9776701e8b93f5a104180f93ba`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack with `LLMK_SM120_DPREP_WARPS=3` and CUDA-kernel grad-zero restored; restored `train_gpt2cu` sha256 is `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The composition is correctness-clean, but it is `28.407 ms` slower than `new-goal.md`, `9.823 ms` slower than the latest direct selected-stack rerun, `16.903 ms` slower than runtime-grad-zero x10 first3, and `11.497 ms` slower than dprep=2 alone. Do not x10-gate or promote.

### `runtime_grad_zero_disable_profiler_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_disable_profiler_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_runtime_grad_zero_disable_profiler_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_runtime_grad_zero_disable_profiler_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_DISABLE_CUDA_PROFILER" scripts/run_sm120_optimization_round.sh`.
- Rationale: test whether the closest CUDA-runtime grad-zero trainer route benefits from removing profiler start/stop overhead, without changing math kernels or the selected dprep, memory, LayerNorm, and ZeRO settings.
- Build/runtime settings: cuBLASLt GEMM, current promoted attention route with default tiles, CUDA runtime grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_DISABLE_CUDA_PROFILER`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: qkv dInput `TK 1087.30 us / cuBLASLt 1013.70 us / cuBLAS 1032.68 us`; fcproj dInput+dGeLU `TK 1834.56 us / cuBLASLt fused 1861.21 us`; attention `783.121/2741.067 us` forward/backward; LayerNorm `137.486/273.346/273.054 us` forward/fused-residual/backward; runtime grad memset `CUDA runtime 148.771 us / CUDA kernel 149.626 us`; AdamW update `1782.624 us`; encoder forward `76.649 us`.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2497.87, 2493.19, 2499.57 ms.
- Trainer average: 2496.375442 ms.
- Visible first-three average: 2496.876667 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Candidate binary hash: `train_gpt2cu` sha256 `6a90766385fdbc0f58631538b0d98be858c806d2fb5e6667af4f177deb695a32`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack with CUDA-kernel grad-zero and profiler calls restored; restored `train_gpt2cu` sha256 is `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The composition is correctness-clean, but it is `27.034 ms` slower than `new-goal.md`, `8.450 ms` slower than the latest direct selected-stack rerun, `15.530 ms` slower than runtime-grad-zero x10 first3, and `12.733 ms` slower than profiler-disable alone. Do not x10-gate or promote.

### `precompute_disable_profiler_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_precompute_disable_profiler_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_precompute_disable_profiler_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_precompute_disable_profiler_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW -DLLMK_DISABLE_CUDA_PROFILER" scripts/run_sm120_optimization_round.sh`.
- Rationale: compose precomputed AdamW grad-scale with disabled CUDA profiler calls because both were plausible near-current individual rows.
- Build/runtime settings: cuBLASLt GEMM, current promoted attention route with default tiles, CUDA-kernel grad-zero, Torch C++ dresidual-zero, precomputed device AdamW scalar, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_DISABLE_CUDA_PROFILER`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: qkv dInput `TK 1088.11 us / cuBLASLt 1016.07 us / cuBLAS 1017.33 us`; fcproj dInput+dGeLU `TK 1790.53 us / cuBLASLt fused 1844.12 us`; attention `784.705/2742.729 us` forward/backward; LayerNorm `137.279/276.462/270.760 us` forward/fused-residual/backward; runtime grad memset `CUDA runtime 149.570 us / CUDA kernel 149.747 us`; AdamW update `1832.470 us`; encoder forward `85.732 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = precomputed device AdamW scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2498.00, 2493.72, 2498.79 ms.
- Trainer average: 2496.254683 ms.
- Visible first-three average: 2496.836667 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609930`.
- Candidate binary hash: `train_gpt2cu` sha256 `157c164e05d831782b7de5c5b6793814b1c1e069ef4c8a93312950ac95bec943`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack with host-scalar AdamW grad scale and profiler calls restored; restored `train_gpt2cu` sha256 is `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The composition is correctness-clean, but it is `26.994 ms` slower than `new-goal.md`, `8.410 ms` slower than the latest selected direct rerun, and slower than both component rows. Do not x10-gate or promote.

### `zero0_disable_profiler_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_zero0_disable_profiler_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_zero0_disable_profiler_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_zero0_disable_profiler_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=0 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_DISABLE_CUDA_PROFILER" scripts/run_sm120_optimization_round.sh`.
- Rationale: compose single-GPU no-ZeRO training with disabled CUDA profiler calls because both were plausible near-current individual rows.
- Build/runtime settings: cuBLASLt GEMM, current promoted attention route with default tiles, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, `LLMK_DISABLE_CUDA_PROFILER`, and `-z 0`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: qkv dInput `TK 1091.36 us / cuBLASLt 1037.70 us / cuBLAS 1014.93 us`; fcproj dInput+dGeLU `TK 1786.04 us / cuBLASLt fused 1813.26 us`; attention `784.447/2746.749 us` forward/backward; LayerNorm `134.967/275.367/270.386 us` forward/fused-residual/backward; runtime grad memset `CUDA runtime 149.309 us / CUDA kernel 150.099 us`; AdamW update `1792.301 us`; encoder forward `85.834 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 0`.
- Step timings: 2497.49, 2492.89, 2500.05 ms.
- Trainer average: 2496.471643 ms.
- Visible first-three average: 2496.810000 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609930`.
- Candidate binary hash: `train_gpt2cu` sha256 `5499204f75f94630e7ca91435ce72c0cedf8414c23bab8d0aec64086f280aa4c`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack with profiler calls and ZeRO stage 1 restored; restored `train_gpt2cu` sha256 is `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The composition is correctness-clean, but it is `26.967 ms` slower than `new-goal.md`, `8.383 ms` slower than the latest selected direct rerun, and slower than both component rows. Do not x10-gate or promote.

### `direct_train_sm120_codex_rerun_after_user_fast_20260522`

- Command: `./train-sm120.sh`.
- Rationale: rerun the exact selected script after the user reported a faster `2461-2465 ms` band and noted earlier measurements may have hit a busy GPU window.
- Build/runtime settings: selected cuBLASLt-backed trainer route, current promoted attention route with default tiles, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, script-default `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2491.74, 2486.59, 2489.38, 2492.17, 2498.22, 2499.68, 2503.29, 2505.26, 2505.75, 2507.99 ms.
- Trainer average: 2498.704116 ms.
- Visible x10 average: 2498.007000 ms.
- Visible first-three average: 2489.236667 ms.
- Visible first-five average: 2491.620000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Active binary hash: `train_gpt2cu` sha256 `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`.
- Cleanup: removed generated `model_00000010.bin` and `state_00000010_00000.bin`; retained `log124M/5090_S`, `main.log`, and `DONE_00000010`.
- Decision: reproduction miss. This run is `19.394 ms` slower than the `new-goal.md` first-three target, `27.787 ms` slower than the user's pasted first-three band, and `29.384 ms` slower than the user's pasted first-five band, so it does not justify changing the selected kernel stack.

### `matmul_dbias768_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_matmul_dbias768_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_matmul_dbias768_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS=-DLLMK_SM120_BIAS_BLOCK_SIZE=768 scripts/run_sm120_optimization_round.sh`.
- Rationale: close the matmul backward-bias reduction block-size sweep. The selected stack uses the default 512-thread reduction, and 256/1024 were already rejected; 768 matches the older upstream-style heuristic and had not been tested in the promoted stack.
- Build/runtime settings: selected promoted stack plus `LLMK_SM120_BIAS_BLOCK_SIZE=768`; cuBLASLt GEMM, current promoted attention route, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: qkv dInput `TK 1087.69 us / cuBLASLt 1036.02 us / cuBLAS 1012.37 us`; fcproj dInput+dGeLU `TK 1822.47 us / cuBLASLt fused 1847.85 us`; attention `783.308/2741.627 us` forward/backward; LayerNorm `136.594/276.261/270.834 us` forward/fused-residual/backward; runtime bias-grad reduce `22.790/200.814/245.667 us` for OC `768/2304/3072`; runtime grad memset `CUDA runtime 149.253 us / CUDA kernel 149.958 us`; AdamW update `1792.147 us`; encoder forward `85.944 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2499.60, 2494.41, 2501.38 ms.
- Trainer average: 2497.894287 ms.
- Visible first-three average: 2498.463333 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609926`.
- Candidate binary hash: `train_gpt2cu` sha256 `8fa6406b868b634e484bef219064b1578bb9727dd27c430ac0adc1838ad83c11`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack without `LLMK_SM120_BIAS_BLOCK_SIZE`; restored `train_gpt2cu` sha256 is `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The 768-thread dbias reduction variant is `28.620 ms` slower than `new-goal.md` first-three, `9.227 ms` slower than the latest selected direct rerun, `9.197 ms` slower than dbias1024, and `3.033 ms` slower than dbias256. Keep the default 512-thread dbias reduction block.

### `direct_train_sm120_codex_rerun_after_user_busy_20260522`

- Artifact summary: `scratch/sm120_rounds/direct_train_sm120_codex_rerun_after_user_busy_20260522/train-sm120-summary.md`
- Command: `bash train-sm120.sh`.
- Rationale: rerun the exact selected script after the user reported their earlier GPU was busy and then observed a faster first-five band around `2462 ms`.
- Build/runtime settings: selected cuBLASLt-backed trainer route, current promoted attention route with default tiles, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, script-default `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2489.65, 2485.15, 2487.15, 2495.33, 2499.15, 2498.13, 2502.87, 2503.86, 2505.36, 2507.66 ms.
- Trainer average: 2498.294274 ms.
- Visible x10 average: 2497.431000 ms.
- Visible first-three average: 2487.316667 ms.
- Visible first-five average: 2491.286000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Active binary hash: `train_gpt2cu` sha256 `a60e97a69daf57498ad339a2d8caf0f8c71b271d336a8ed0a8aace289aabb9be`.
- Checkpoints: direct script wrote and retained `log124M/5090_S/model_00000010.bin` and `log124M/5090_S/state_00000010_00000.bin`.
- Decision: reproduction miss. This run is `17.474 ms` slower than the `new-goal.md` first-three target, `25.867 ms` slower than the user's pasted first-three band, and `29.050 ms` slower than the user's pasted first-five band. It is close to the recent Codex direct-script band, so this does not justify changing the selected kernel stack.

### `global_norm_block768_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_global_norm_block768_x3_20260522`
- Trainer output: `log124M/5090_S_codex_sm120_global_norm_block768_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_global_norm_block768_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS=-DLLMK_SM120_GLOBAL_NORM_BLOCK_SIZE=768 scripts/run_sm120_optimization_round.sh`.
- Rationale: close the untested middle block size in the global-norm sweep. The selected stack uses the default 512-thread block, while 256 and 1024 already had rejection evidence.
- Build/runtime settings: selected promoted stack plus `LLMK_SM120_GLOBAL_NORM_BLOCK_SIZE=768`; cuBLASLt GEMM, current promoted attention route, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Focused correctness: `test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm` passed on RTX 5090.
- Native benchmark highlights: qkv dInput `TK 1103.97 us / cuBLASLt 1020.52 us / cuBLAS 1012.16 us`; fcproj dInput+dGeLU `TK 1786.31 us / cuBLASLt fused 1839.00 us`; attention `785.353/2738.536 us` forward/backward; LayerNorm `138.748/274.923/272.723 us` forward/fused-residual/backward; global norm `185.144 us`; runtime grad memset `CUDA runtime 149.496 us / CUDA kernel 150.070 us`; AdamW update `1787.427 us`; encoder forward `76.243 us`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2499.17, 2492.22, 2499.07 ms.
- Trainer average: 2495.644927 ms.
- Visible first-three average: 2496.820000 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609931`.
- Candidate binary hash: `train_gpt2cu` sha256 `49efaa572bb185af50706b627f0e2ff37bb558c3c61a634b821d700141eb0a5f`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack without `LLMK_SM120_GLOBAL_NORM_BLOCK_SIZE`; restored `train_gpt2cu` sha256 is `e2c853701f50d71e1540a520156e2bb6c7046654afb68e9bf37892cd049597ed`, all nine focused CUDA smokes passed, and a one-step startup probe confirmed the selected backend mix.
- Decision: rejected. The 768-thread global-norm block variant is `26.977 ms` slower than `new-goal.md` first-three and `9.503 ms` slower than the latest direct selected-stack rerun. Keep the default 512-thread global-norm block.

### `direct_train_sm120_e2c_restored_rerun_x10_20260522`

- Artifact summary: `scratch/sm120_rounds/direct_train_sm120_e2c_restored_rerun_x10_20260522/train-sm120-summary.md`
- Command: `bash train-sm120.sh`.
- Rationale: rerun the exact selected script after restoring from the global-norm candidate, because the restored selected-source build produced `train_gpt2cu` hash `e2c85370...` while preserving the intended backend startup mix.
- Build/runtime settings: selected cuBLASLt-backed trainer route, current promoted attention route with default tiles, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, script-default `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2492.34, 2486.46, 2488.54, 2490.47, 2498.35, 2500.40, 2501.46, 2504.56, 2507.65, 2506.68 ms.
- Trainer average: 2498.284976 ms.
- Visible x10 average: 2497.691000 ms.
- Visible first-three average: 2489.113333 ms.
- Visible first-five average: 2491.232000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Active binary hash: `train_gpt2cu` sha256 `e2c853701f50d71e1540a520156e2bb6c7046654afb68e9bf37892cd049597ed`.
- Checkpoints: direct script wrote and retained `log124M/5090_S/model_00000010.bin` and `log124M/5090_S/state_00000010_00000.bin`.
- Decision: reproduction miss. The restored selected-source binary is `19.270 ms` slower than `new-goal.md` first-three, `27.663 ms` slower than the user's pasted first-three band, and `28.996 ms` slower than the user's pasted first-five band. Rebuilding changed the hash but did not recover the user's faster band.

### `direct_train_sm120_e2c_user_busy_rerun2_x10_20260522`

- Artifact summary: `scratch/sm120_rounds/direct_train_sm120_e2c_user_busy_rerun2_x10_20260522/train-sm120-summary.md`
- Command: `bash train-sm120.sh`.
- Rationale: rerun the exact selected script again after the user reported their own rerun showed `~2462 ms` first-five timing and noted that the GPU may have been busy during prior Codex runs.
- Build/runtime settings: selected cuBLASLt-backed trainer route, current promoted attention route with default tiles, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, script-default `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2488.09, 2483.44, 2487.56, 2489.60, 2491.58, 2499.00, 2500.83, 2503.03, 2503.78, 2505.35 ms.
- Trainer average: 2496.017721 ms.
- Visible x10 average: 2495.226000 ms.
- Visible first-three average: 2486.363333 ms.
- Visible first-five average: 2488.054000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Active binary hash: `train_gpt2cu` sha256 `e2c853701f50d71e1540a520156e2bb6c7046654afb68e9bf37892cd049597ed`.
- Checkpoints: direct script wrote and retained `log124M/5090_S/model_00000010.bin` and `log124M/5090_S/state_00000010_00000.bin`.
- Decision: reproduction miss, but a better no-contention control than the immediately previous Codex rerun. This run is `16.520 ms` slower than `new-goal.md` first-three, `24.913 ms` slower than the user's pasted first-three band, and `25.818 ms` slower than the user's pasted first-five band, so it still does not justify changing the selected kernel stack.

### `tk_dgelu_approx_tanh_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_tk_dgelu_approx_tanh_x3_20260522`
- Decision note: `scratch/sm120_rounds/codex_sm120_tk_dgelu_approx_tanh_x3_20260522/decision.md`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_tk_dgelu_approx_tanh_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP -DLLMK_SM120_APPROX_DGELU_TANH=1" scripts/run_sm120_optimization_round.sh`.
- Rationale: close the remaining trainer-callable TK fused dGELU internal knob. Focused matmul rows have sometimes shown the TK fcproj dInput+dGeLU path close to or faster than cuBLASLt fused dGELU, while exact TK dGELU integration was already rejected end-to-end; this trial checked whether the approximate tanh derivative could pass correctness before any trainer timing.
- Build/runtime settings: selected stack plus `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` and `LLMK_SM120_APPROX_DGELU_TANH=1`; cuBLASLt GEMM remains enabled for the rest of the trainer, with CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Correctness: `test_matmul` passed 9/10 rows, but failed `dInp backward fused dGELU (GPT-2 fcproj route)` with `max abs diff = 0.5000` at tolerance `0.50`.
- Candidate binary hash: `train_gpt2cu` sha256 `549736f9558be4bee7a6f28fee305f21491c5a3a51ba22b7c6e17a8d9b2ae084`.
- Benchmarks/training: not accepted and not run after the failing correctness gate.
- Cleanup: no `log124M/5090_S_codex_sm120_tk_dgelu_approx_tanh_x3_20260522` training output directory was created.
- Restore: rebuilt the selected default stack without the candidate dGELU flags; restored `train_gpt2cu` sha256 is `4374a593bdc123692c38f28c92951abbe914c4f56f96a9fd4e140fa3c4b119da`, and all nine focused CUDA smokes passed after restoration.
- Decision: correctness rejected. Do not promote or x10-gate this candidate unless the GPT-2 fcproj fused dGELU parity gap is fixed first.

### `direct_train_sm120_4374_post_tk_dgelu_restore_x10_20260522`

- Artifact summary: `scratch/sm120_rounds/direct_train_sm120_4374_post_tk_dgelu_restore_x10_20260522/train-sm120-summary.md`
- Command: `bash train-sm120.sh`.
- Rationale: record the exact selected script after restoring from the correctness-rejected TK approximate dGELU candidate, because the selected binary hash changed to `4374a593...` after the clean restore build.
- Build/runtime settings: selected cuBLASLt-backed trainer route, current promoted attention route with default tiles, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, script-default `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2490.11, 2484.31, 2487.35, 2491.20, 2495.80, 2501.34, 2504.18, 2507.46, 2507.20, 2508.05 ms.
- Trainer average: 2498.544534 ms.
- Visible x10 average: 2497.700000 ms.
- Visible first-three average: 2487.256667 ms.
- Visible first-five average: 2489.754000 ms.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Active binary hash: `train_gpt2cu` sha256 `4374a593bdc123692c38f28c92951abbe914c4f56f96a9fd4e140fa3c4b119da`.
- Checkpoints: direct script wrote and retained `log124M/5090_S/model_00000010.bin` and `log124M/5090_S/state_00000010_00000.bin`.
- Decision: reproduction miss. This run is `17.414 ms` slower than `new-goal.md` first-three, `25.807 ms` slower than the user's pasted first-three band, and `27.518 ms` slower than the user's pasted first-five band. It confirms the selected stack was restored, but does not justify changing the selected kernel stack.

### `cublaslt_ws64_x3_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_cublaslt_ws64_x3_20260522`
- Command: `CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_LABEL=codex_sm120_cublaslt_ws64_x3_20260522 MAX_STEPS=3 TRAIN_ZERO_STAGE=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_CUBLASLT_WORKSPACE_MB=64" scripts/run_sm120_optimization_round.sh`.
- Rationale: close the smaller cuBLASLt workspace knob. The larger workspace/heuristic trial was already rejected, but a 64 MiB workspace could have changed algorithm selection and memory pressure in the opposite direction.
- Build/runtime settings: selected stack plus `LLMK_SM120_CUBLASLT_WORKSPACE_MB=64`, with CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Correctness: all required smoke tests passed (`test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`).
- Benchmarks: native benchmark phase passed and validator accepted `95` benchmark rows.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Step timings: 2498.25, 2494.75, 2501.39 ms.
- Trainer average: 2498.070717 ms.
- Visible first-three average: 2498.130000 ms.
- Losses: initial val loss `11.033154`, final val loss `10.609911`.
- Candidate binary hash: `train_gpt2cu` sha256 `733f6c217b9b113e57f7871d0ad58bfc1299e54cf462e8702d996af8b9ae3f23`.
- Cleanup: harness removed generated checkpoint files; output dir retains `DONE_00000003` and `main.log`.
- Restore: rebuilt the selected default stack without the workspace candidate flag; restored `train_gpt2cu` sha256 is `cef1ac4a06faee188cc46a98317f3b36b3002fffb7bdc3be10842e095797636c`, and all nine focused CUDA smokes passed after restoration.
- Decision: rejected. The 64 MiB cuBLASLt workspace variant is `28.287 ms` slower than `new-goal.md` first-three, `11.767 ms` slower than the latest direct selected-stack rerun, and `36.680 ms` slower than the user's pasted first-three band. Keep the default cuBLASLt workspace setting.

### `direct_train_sm120_cef_user_rerun3_x10_20260522`

- Artifact summary: `scratch/sm120_rounds/direct_train_sm120_cef_user_rerun3_x10_20260522/train-sm120-summary.md`
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

### `direct_train_sm120_cef_codex_idle_rerun4_x10_20260522`

- Artifact summary: `scratch/sm120_rounds/direct_train_sm120_cef_codex_idle_rerun4_x10_20260522/train-sm120-summary.md`
- Command: `./train-sm120.sh`.
- Rationale: rerun the exact selected script after the user pasted a faster no-contention sample and suggested the prior Codex run may have hit a busy GPU.
- Build/runtime settings: selected cuBLASLt-backed trainer route, packed-QKV TK attention, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, script-default `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `B=64`, `T=1024`, `total_batch_size=524288`, `grad_accum_steps=8`, `gelu_fusion=1`, `grad_zero_backend=CUDA kernel`, `dresidual_zero_backend=Torch C++`, `grad_scale_backend=host scalar`, and `zero_stage=1`.
- GPU state: post-run `nvidia-smi` reported no running GPU processes, P8 idle state, `798 MiB / 32607 MiB`, and `1%` GPU utilization.
- Step timings: `2493.99`, `2488.06`, `2489.74`, `2495.45`, `2499.81`, `2501.69`, `2505.13`, `2506.93`, `2509.96`, `2509.84 ms`.
- Trainer average: `2500.735177 ms`.
- Visible x10 average: `2500.060000 ms`.
- Visible first-three average: `2490.596667 ms`.
- Visible first-five average: `2493.410000 ms`.
- User pasted comparison sample: first five steps `2464.71`, `2456.65`, `2462.99`, `2461.49`, `2465.34 ms`; first-five average `2462.236000 ms`.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Active binary hash: `train_gpt2cu` sha256 `cef1ac4a06faee188cc46a98317f3b36b3002fffb7bdc3be10842e095797636c`.
- Decision: reproduction miss. This run is `31.174 ms` slower than the user's pasted first-five average and does not justify a kernel-stack change.

### `direct_train_sm120_cef_codex_idle_rerun5_x10_20260522`

- Artifact summary: `scratch/sm120_rounds/direct_train_sm120_cef_codex_idle_rerun5_x10_20260522/train-sm120-summary.md`
- Command: `./train-sm120.sh`.
- Rationale: second immediate selected-script rerun after confirming the GPU had no active processes.
- Build/runtime settings: selected cuBLASLt-backed trainer route, packed-QKV TK attention, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, script-default `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`, and `-z 1`.
- Startup confirmed `B=64`, `T=1024`, `total_batch_size=524288`, `grad_accum_steps=8`, `gelu_fusion=1`, `grad_zero_backend=CUDA kernel`, `dresidual_zero_backend=Torch C++`, `grad_scale_backend=host scalar`, and `zero_stage=1`.
- Step timings: `2501.68`, `2499.65`, `2502.26`, `2506.16`, `2508.45`, `2511.15`, `2511.90`, `2513.34`, `2515.75`, `2517.51 ms`.
- Trainer average: `2509.575870 ms`.
- Visible x10 average: `2508.785000 ms`.
- Visible first-three average: `2501.196667 ms`.
- Visible first-five average: `2503.640000 ms`.
- User pasted comparison sample: first-five average `2462.236000 ms`.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Active binary hash: `train_gpt2cu` sha256 `cef1ac4a06faee188cc46a98317f3b36b3002fffb7bdc3be10842e095797636c`.
- Decision: reproduction miss and slower than the prior immediate rerun. This points to run-to-run/clock-runtime variability in the same selected stack, not to a different active kernel mix.

### `batch_shape_sweep_20260522`

- Artifact summary: `scratch/sm120_rounds/batch_shape_sweep_20260522/summary.md`
- Rationale: test whether the reported `estimated maximum batch size: 73` can improve training throughput by increasing resident microbatch size and reducing or reshaping gradient accumulation. This is a training-shape sweep on the selected kernel stack, not a kernel/provider change.
- Selected backend mix for all runs: cuBLASLt-backed trainer route, packed-QKV TK attention, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, ZeRO stage 1, BF16, and `gelu_fusion = 1`.
- `B=73`, `grad_accum_steps=7`, `total_batch_size=523264`: device usage `32397 MiB / 32606 MiB`, steps `2823.95, 2832.62, 2835.00, 2844.93, 2868.52 ms`, visible x5 average `2841.004000 ms`, normalized throughput `184182.8 tok/s`.
- `B=70`, `grad_accum_steps=7`, `total_batch_size=501760`: device usage `31203 MiB / 32606 MiB`, steps `2395.42, 2382.68, 2393.00 ms`, visible first-three average `2390.366667 ms`, normalized throughput `209909.2 tok/s`.
- `B=65`, `grad_accum_steps=8`, `total_batch_size=532480`: device usage `29215 MiB / 32606 MiB`, steps `2547.52, 2541.77, 2547.06 ms`, visible first-three average `2545.450000 ms`, normalized throughput `209188.9 tok/s`.
- `B=71`, `grad_accum_steps=7`, `total_batch_size=508928`: x3 device usage `31601 MiB / 32606 MiB`, x3 steps `2423.41, 2417.18, 2421.05 ms`, normalized throughput `210252.5 tok/s`; x10 gate steps `2421.76, 2415.94, 2422.27, 2434.19, 2435.77, 2438.00, 2431.10, 2439.64, 2443.31, 2443.45 ms`, visible x10 average `2432.543000 ms`, normalized throughput `209216.4 tok/s`.
- `B=72`, `grad_accum_steps=7`, `total_batch_size=516096`: device usage `31999 MiB / 32606 MiB`, steps `2457.74, 2453.53, 2458.33 ms`, visible first-three average `2456.533333 ms`, normalized throughput `210089.5 tok/s`.
- `B=68`, `grad_accum_steps=8`, `total_batch_size=557056`: x3 device usage `30409 MiB / 32606 MiB`, x3 steps `2651.27, 2645.59, 2649.39 ms`, normalized throughput `210309.0 tok/s`; x10 gate steps `2647.59, 2643.03, 2648.98, 2653.97, 2664.64, 2665.71, 2669.43, 2670.59, 2672.21, 2668.85 ms`, visible x10 average `2660.500000 ms`, normalized throughput `209380.2 tok/s`.
- Reference: latest direct selected-stack rerun `direct_train_sm120_cef_user_rerun3_x10_20260522`, `B=64`, `grad_accum_steps=8`, `total_batch_size=524288`, visible x10 average `2498.807000 ms`, normalized throughput `209815.3 tok/s`, visible first-three `2489.346667 ms`.
- Decision: rejected as a training-default change. `B=73` is memory-saturated and much slower; `B=65` is slower; `B=70/GA=7` only lowers milliseconds per step by processing fewer tokens and is effectively flat on normalized throughput; `B=71/GA=7` failed the x10 gate; `B=72/GA=7` is slower than `B=71`; `B=68/GA=8` also failed the x10 gate. No significant normalized training-speed improvement was found.

### `current_optional_refresh_and_libtorch_grad_zero_gate_20260522`

- Benchmark artifact: `scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522`.
- Rationale: refresh the optional stack scorecard under the current source/runtime state before selecting the next combination. The prior current-selection audit had no active promotion candidates, but it was based on older rounds.
- Refresh command: `RUN_LABEL=codex_sm120_current_optional_refresh_20260522 RUN_TRAINING=0 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=1 RUN_STACK_PROBE=1 RUN_ARTIFACT_VALIDATOR=1 BUILD_JOBS=4 CONDA_PREFIX=/home/adam/miniconda3/envs/llm-kittens PYTHON_BIN=/home/adam/miniconda3/envs/llm-kittens/bin/python scripts/run_sm120_optimization_round.sh`.
- Refresh result: validator accepted `210` benchmark rows, `43/43` Torch objective rows, `5/5` LibTorch runtime rows, `5/5` LibTorch parity rows, and the LibTorch trainer-link probe.
- Active refreshed candidates: Torch/Torch C++ grad-buffer memset (`grad_elems=124475904`) and Torch native LayerNorm forward (`N=65536 C=768`). Both are low-priority and non-promotable as-is: the LayerNorm row lacks trainer-compatible saved mean/rstd state, while the grad-zero row has an existing LibTorch C++ trainer route that can be re-gated.
- Candidate x3 artifact: `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_current_refresh_x3_20260522`.
- Candidate x3 settings: selected cuBLASLt/TK/CUDA stack plus `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and ZeRO stage 1.
- Candidate x3 startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `zero_stage = 1`.
- Candidate x3 step timings: `2501.23`, `2498.51`, `2502.75 ms`; trainer average `2500.630260 ms`; visible first-three average `2500.830000 ms`.
- Same-session selected-control artifact: `scratch/sm120_rounds/codex_sm120_selected_restore_after_libtorch_grad_current_x3_20260522`.
- Same-session selected-control startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `zero_stage = 1`.
- Same-session selected-control step timings: `2506.60`, `2503.57`, `2505.07 ms`; trainer average `2504.322290 ms`; visible first-three average `2505.080000 ms`.
- Candidate x10 artifact: `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_current_refresh_x10_20260522`.
- Candidate x10 step timings: `2497.53`, `2490.42`, `2497.78`, `2502.52`, `2504.86`, `2506.18`, `2507.94`, `2511.05`, `2513.33`, `2515.33 ms`.
- Candidate x10 trainer average: `2505.489906 ms`; visible x10 average `2504.694000 ms`; visible first-three average `2495.243333 ms`; visible first-five average `2498.622000 ms`.
- Losses: candidate x3 `11.033154 -> 10.609911`; selected-control x3 `11.033154 -> 10.609911`; candidate x10 `11.033154 -> 9.483727`.
- Cleanup: harness deleted generated round checkpoints. Old base `log124M/5090_S/model_00000010.bin` and `log124M/5090_S/state_00000010_00000.bin` were removed while preserving the base log directory.
- Restore: rebuilt selected stack with CUDA-kernel grad-zero plus Torch C++ dresidual-zero. Active `train_gpt2cu` sha256 after restore is `ad9c7d14c0b30ae74648342389e1b99c83043d0afd5a34d63a8ba1b37f864eec`.
- Decision: rejected. The refreshed Torch/LibTorch memset edge was real enough to test in the trainer, and the candidate beat the same-session selected-control x3 by `3.692 ms` trainer average, but its x10 gate averaged `2505.490 ms`, slower than the selected x10 references, `new-goal.md`, and the user-pasted fast band. Keep CUDA-kernel grad-zero selected.

### `selected_current_audit_x10_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_selected_current_audit_x10_20260522`.
- Command: `RUN_LABEL=codex_sm120_selected_current_audit_x10_20260522 MAX_STEPS=10 TRAIN_ZERO_STAGE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=1 RUN_ARTIFACT_VALIDATOR=1 RUN_CURRENT_SELECTION_AUDIT=0 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 CONDA_PREFIX=/home/adam/miniconda3/envs/llm-kittens PYTHON_BIN=/home/adam/miniconda3/envs/llm-kittens/bin/python scripts/run_sm120_optimization_round.sh`.
- Rationale: refresh the current selected-stack x10 baseline with full correctness, native benchmark, stack-probe, and artifact validation after the user's faster no-contention sample and after the 2026-05-22 optional-stack refresh.
- Build/runtime settings: selected cuBLASLt-backed trainer route, packed-QKV TK attention, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, ZeRO stage 1, BF16, `gelu_fusion=1`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`.
- Correctness/benchmarks/audit: all nine focused CUDA smokes passed, native benchmark validation accepted `95` rows, and the regenerated current audit passed `132` checks with `43/43` trainer rows, full optional Torch/Triton/cuDNN/CuTeDSL/LibTorch coverage, and `0` active promotion candidates.
- Step timings: `2504.31`, `2498.49`, `2502.45`, `2513.12`, `2511.21`, `2509.69`, `2513.11`, `2516.85`, `2517.20`, and `2520.51 ms`.
- Trainer average: `2511.402872 ms`; visible x10 average `2510.694000 ms`; visible first-three average `2501.750000 ms`; visible first-five average `2505.916000 ms`.
- Native benchmark highlights: fcproj `dInp+dGeLU` TK/cuBLASLt-fused `1773.45/1841.50 us`, LayerNorm C=768 forward/fused/backward `138.314/275.438/270.512 us`, runtime grad memset CUDA runtime/kernel `149.554/149.502 us`, and AdamW update `1828.384 us`.
- Binary hash after the control build: `train_gpt2cu` sha256 `9d329e98b96bd9e2a9e2a5c7dd4686b04ddba5ae4c853c69e85583572954751c`.
- Decision: control only. This proves the selected stack is active but remains a slow-band reproduction miss versus the user's pasted first-five average `2462.236 ms`; it is not a promotion or completion signal.

### `combo_libtorch_grad_zero_all_native_winners_x10_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_all_native_winners_x10_20260522`.
- Command: `RUN_LABEL=codex_sm120_combo_libtorch_grad_zero_all_native_winners_x10_20260522 MAX_STEPS=10 TRAIN_ZERO_STAGE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 RUN_CURRENT_SELECTION_AUDIT=0 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=1 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_ATTPROJ -DLLMK_SM120_USE_CUBLAS_DINP_FC -DLLMK_SM120_USE_TK_FUSED_DGELU_DINP -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1 -DLLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024" CONDA_PREFIX=/home/adam/miniconda3/envs/llm-kittens PYTHON_BIN=/home/adam/miniconda3/envs/llm-kittens/bin/python scripts/run_sm120_optimization_round.sh`.
- Rationale: x10-gate the best previous ungated composed route: LibTorch grad-zero plus direct-cuBLAS dInput for attention projection/MLP-up, TK fused dGELU dInput for MLP projection, one-block LayerNorm backward, and wide bias-add. This checks whether the earlier x3 near-miss survives a real trainer stability gate under the current runtime band.
- Startup confirmed `grad_zero_backend = Torch C++`, `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Correctness/benchmarks: all nine focused CUDA smokes passed and artifact validation accepted `95` native benchmark rows.
- Native benchmark highlights: fcproj `dInp+dGeLU` TK/cuBLASLt-fused `1741.99/1853.93 us`, fc dInput cuBLASLt/cuBLAS `1350.42/1333.49 us`, attproj dInput cuBLASLt/cuBLAS `367.30/365.56 us`, LayerNorm C=768 forward/fused/backward `137.429/275.929/269.623 us`, and runtime grad memset CUDA runtime/kernel `148.032/150.573 us`.
- Step timings: `2504.49`, `2508.37`, `2520.56`, `2513.75`, `2516.09`, `2518.59`, `2520.37`, `2523.73`, `2526.29`, and `2528.73 ms`.
- Trainer average: `2519.609557 ms`; visible x10 average `2518.097000 ms`; visible first-three average `2511.140000 ms`; visible first-five average `2512.652000 ms`.
- Candidate binary hash before restore: captured in `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_all_native_winners_x10_20260522/round-manifest.json`.
- Restore: rebuilt the selected stack with CUDA-kernel grad-zero plus Torch C++ dresidual-zero. Active `train_gpt2cu` sha256 after restore is `d37c652bfbc4f95e24d47247ea25619560d98e59a1388140718a37c09571b6c5`.
- Decision: rejected. The component microbenchmarks still show wins, but the x10 trainer gate is `8.207 ms` slower than the current selected x10 control and far slower than the user's `2462.236 ms` first-five band. Do not promote this composition.

### `cublas_dinp_fcproj_current_rerun_x10_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522`.
- Command: `RUN_LABEL=codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522 MAX_STEPS=10 TRAIN_ZERO_STAGE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 RUN_CURRENT_SELECTION_AUDIT=0 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLAS_DINP_FCPROJ" CONDA_PREFIX=/home/adam/miniconda3/envs/llm-kittens PYTHON_BIN=/home/adam/miniconda3/envs/llm-kittens/bin/python scripts/run_sm120_optimization_round.sh`.
- Rationale: rerun the previously noisy fcproj direct-cuBLAS dInput candidate under the current source/runtime state after the user reported a faster less-contended training sample.
- Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Correctness/benchmarks: all nine focused CUDA smokes passed and artifact validation accepted `95` native benchmark rows.
- Native benchmark highlights: fcproj `dInp+dGeLU` TK/cuBLASLt-fused `1818.53/1831.64 us`, fcproj dInput TK/cuBLASLt/cuBLAS `1480.32/1437.77/1371.00 us`, runtime grad memset CUDA runtime/kernel `156.770/161.787 us`, and AdamW update `1864.371 us`.
- Step timings: `2524.47`, `2517.24`, `2521.78`, `2525.43`, `2526.93`, `2531.79`, `2530.33`, `2535.65`, `2533.24`, and `2538.55 ms`.
- Trainer average: `2528.994269 ms`; visible x10 average `2528.541000 ms`; visible first-three average `2521.163333 ms`; visible first-five average `2523.170000 ms`.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Restore: rebuilt the selected stack with CUDA-kernel grad-zero plus Torch C++ dresidual-zero before rerunning the direct script. Active `train_gpt2cu` sha256 after restore and script rerun is `2032b2086406ee051fa8f18a5ae2c716c70e6d8165f4bc5383fb24e7c7657ac1`.
- Decision: rejected. The current rerun is `17.591 ms` slower than the current selected x10 control and still much slower than the user's pasted fast band.

### `direct_train_sm120_after_fcproj_restore_x10_20260522`

- Artifact summary: `scratch/sm120_rounds/direct_train_sm120_after_fcproj_restore_x10_20260522/train-sm120-summary.md`.
- Command: `./train-sm120.sh`.
- Rationale: run the exact selected wrapper after restoring from the rejected fcproj direct-cuBLAS candidate, matching the user's latest request to rerun the script because the previous Codex run may have hit GPU contention.
- Startup confirmed `B=64`, `T=1024`, `total_batch_size=524288`, `grad_accum_steps=8`, `gelu_fusion=1`, `grad_zero_backend=CUDA kernel`, `dresidual_zero_backend=Torch C++`, `grad_scale_backend=host scalar`, and `zero_stage=1`.
- Step timings: `2517.96`, `2517.30`, `2518.46`, `2524.09`, `2589.31`, `2693.36`, `2650.45`, `2549.43`, `2554.61`, and `2555.64 ms`.
- Trainer average: `2572.516229 ms`; visible x10 average `2567.061000 ms`; visible first-three average `2517.906667 ms`; visible first-five average `2533.424000 ms`.
- User pasted comparison sample: first-five average `2462.236000 ms`.
- Losses: initial val loss `11.033154`, final val loss `9.483727`.
- Post-run GPU snapshot: `P8`, `39 C`, `1%` GPU utilization, `2063 MiB / 32607 MiB`, graphics clock `367 MHz`, memory clock `405 MHz`.
- Checkpoints: direct script wrote and retained `log124M/5090_S/model_00000010.bin` and `log124M/5090_S/state_00000010_00000.bin`.
- Decision: reproduction miss. The exact wrapper is using the selected kernel mix, but this run landed in a slower band with step 5-7 spikes; it is runtime/clock/contender evidence, not a reason to promote a different kernel stack.

### `optional_refresh_current2_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522`.
- Command: `RUN_LABEL=codex_sm120_optional_refresh_current2_20260522 RUN_TRAINING=0 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=1 RUN_STACK_PROBE=1 RUN_ARTIFACT_VALIDATOR=1 RUN_LIBTORCH_MATMUL_BENCHMARKS=1 BUILD_JOBS=4 CONDA_PREFIX=/home/adam/miniconda3/envs/llm-kittens PYTHON_BIN=/home/adam/miniconda3/envs/llm-kittens/bin/python scripts/run_sm120_optimization_round.sh`.
- Rationale: refresh all optional backend-stack evidence under the current source/runtime state before selecting another composed trainer candidate.
- Coverage: all nine focused CUDA smokes passed; validator accepted `210` benchmark rows, `43/43` Torch objective rows, `5/5` LibTorch runtime rows, `5/5` LibTorch parity rows, one supplemental LibTorch `gelu_forward` row, the LibTorch trainer-link probe, nine backend stacks, and `168` family-stack rows.
- Native benchmark highlights: attention packed TK `774.667/2706.639 us` forward/backward; LayerNorm C=768 forward/fused/backward `135.130/271.036/267.484 us`; AdamW update `1785.632 us`; classifier loss/full `3898.758/8793.012 us`; grad memset CUDA runtime/kernel `148.589/149.846 us`.
- Torch/LibTorch highlights: separated Torch SDPA remained fast at `556.565/2160.624 us` forward/backward, but packed trainer-layout Torch was slow at `1142.946/4002.704 us`; cuDNN packed was also slower than packed TK at `790.320/2765.123 us`; LibTorch C++ grad memset was `148.206 us`; LibTorch C++ hidden memset was `59.861 us`; Torch BF16-state AdamW was `1198.336 us` but remains a non-equivalent optimizer-state contract.
- Regenerated selection/audit: `scratch/sm120_rounds/current-sm120-selection.{json,md}` now use native round `codex_sm120_selected_current_audit_x10_20260522` plus optional round `codex_sm120_optional_refresh_current2_20260522`; audit passed `132` checks with `43` trainer rows, `9` project Torch-fastest rows, and `0` active promotion candidates.
- Restore: rebuilt the selected trainer stack after the benchmark refresh. Active `train_gpt2cu` sha256 after restore is `e1be7de3d6adbf0f08d50ef4b2b9b377681fa4aa3719e253bc71e70dae2c63e7`.
- Cleanup: removed direct-script checkpoint payloads `log124M/5090_S/model_00000010.bin` and `log124M/5090_S/state_00000010_00000.bin` while preserving `log124M/5090_S`.
- Decision: no trainer candidate selected from this refresh. The only large Torch wins are separated-Q/K/V attention references requiring a layout rewrite; all trainer-layout packed attention routes are slower than packed TK, and the LibTorch memory rows already have x10 trainer-route rejections. Do not run a TinyStories gate until a refreshed row is both trainer-callable and active.

### `runtime_grad_zero_current2_x10_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522`.
- Command: `RUN_LABEL=codex_sm120_runtime_grad_zero_current2_x10_20260522 MAX_STEPS=10 TRAIN_ZERO_STAGE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=1 RUN_CURRENT_SELECTION_AUDIT=0 SM120_FAST_TRAINER=1 SM120_USE_CUDA_KERNEL_GRAD_ZERO=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 CONDA_PREFIX=/home/adam/miniconda3/envs/llm-kittens PYTHON_BIN=/home/adam/miniconda3/envs/llm-kittens/bin/python scripts/run_sm120_optimization_round.sh`.
- Rationale: the refreshed optional-stack round showed the trainer-sized grad memset row was effectively tied, with CUDA runtime slightly ahead of the CUDA-kernel route in the latest refresh. This reruns the trainer-callable CUDA-runtime grad-zero route while keeping the selected Torch C++ dresidual-zero route.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Correctness/benchmarks: all nine focused CUDA smokes passed and artifact validation accepted `95` native benchmark rows.
- Harness step timings: `2470.01`, `2464.87`, `2468.19`, `2473.92`, `2474.82`, `2477.89`, `2479.35`, `2480.94`, `2483.13`, and `2483.18 ms`.
- Harness trainer average: `2476.255973 ms`; visible x10 average `2475.630000 ms`; visible first-three average `2467.690000 ms`; visible first-five average `2470.362000 ms`.
- Exact wrapper command: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh` using the same rebuilt binary.
- Exact wrapper step timings: `2470.71`, `2462.32`, `2466.90`, `2470.85`, `2475.64`, `2476.73`, `2479.63`, `2479.77`, `2480.99`, and `2483.17 ms`.
- Exact wrapper trainer average: `2475.111405 ms`; visible x10 average `2474.671000 ms`; visible first-three average `2466.643333 ms`; visible first-five average `2469.284000 ms`.
- Comparison: the exact wrapper result is `13.951 ms` faster than the promoted direct-script proof (`2489.062124 ms`) and `35.472 ms` faster than the latest selected-stack x10 audit (`2511.402872 ms`). It is still `7.048 ms` slower than the user's pasted `2462.236 ms` first-five sample, so the fast user band is not fully reproduced.
- Candidate binary hash during the x10 reruns: `train_gpt2cu` sha256 `21a34b8db225946d6a483311c40de83b7cb128673f9e5599c6dd1686d06c1336`.
- Post-promotion default rebuild: `make -B -j 4 train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1 SM120_FAST_TRAINER=1 PYTHON_BIN=/home/adam/miniconda3/envs/llm-kittens/bin/python` built without `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO` and produced `train_gpt2cu` sha256 `6ed5e22c034a7def7ddeed8041b7353e0e5bec4fb289e3a86971224d068e3792`.
- Cleanup: removed direct-script checkpoint payloads `log124M/5090_S/model_00000010.bin` and `log124M/5090_S/state_00000010_00000.bin` while preserving `log124M/5090_S`.
- Decision: promote the CUDA-runtime grad-zero default over the CUDA-kernel grad-zero default for SM120 fast trainer builds. Keep Torch C++ dresidual-zero selected; keep LibTorch/Torch grad-zero rejected because its x10 trainer routes are slower.

### `runtime_grad_zero_default_audit_x10_20260522`

- Artifact directory: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522`.
- Command: `RUN_LABEL=codex_sm120_runtime_grad_zero_default_audit_x10_20260522 MAX_STEPS=10 TRAIN_ZERO_STAGE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 BUILD_JOBS=4 RUN_CORRECTNESS=1 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_STACK_PROBE=1 RUN_ARTIFACT_VALIDATOR=1 RUN_CURRENT_SELECTION_AUDIT=0 SM120_FAST_TRAINER=1 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 CONDA_PREFIX=/home/adam/miniconda3/envs/llm-kittens PYTHON_BIN=/home/adam/miniconda3/envs/llm-kittens/bin/python scripts/run_sm120_optimization_round.sh`.
- Rationale: rerun the promoted default with stack-probe enabled so the published current-selection artifacts have complete provenance for the new CUDA-runtime grad-zero default rather than pointing at an older CUDA-kernel-grad-zero control.
- Startup confirmed `grad_zero_backend = CUDA runtime`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, and `zero_stage = 1`.
- Correctness/benchmarks/probe: all nine focused CUDA smokes passed; native benchmark validation accepted `95` rows; stack probe recorded `9` backend stacks and `168` family-stack rows.
- Step timings: `2468.68`, `2461.91`, `2465.47`, `2467.56`, `2472.84`, `2475.15`, `2477.91`, `2478.44`, `2480.29`, and `2480.50 ms`.
- Trainer average: `2473.341915 ms`; visible x10 average `2472.875000 ms`; visible first-three average `2465.353333 ms`; visible first-five average `2467.292000 ms`.
- Exact post-rebuild wrapper proof: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh` on rebuilt binary `6ed5e22c...` averaged `2468.121529 ms`, with visible x10 `2467.682000 ms`, first-three `2461.186667 ms`, and first-five `2463.022000 ms`.
- Active binary hash after the full audit rebuild: `train_gpt2cu` sha256 `4bfe515d8f36aee2a88b63a6fb0229469eca50ce338975ae363ad06bcca99f1c`.
- Exact post-audit wrapper proof: `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh` on active binary `4bfe515d...` averaged `2465.890302 ms`, with visible x10 `2465.083000 ms`, first-three `2458.770000 ms`, and first-five `2460.134000 ms`.
- Regenerated selection/audit: `scratch/sm120_rounds/current-sm120-selection.{json,md}` now uses this native round plus `codex_sm120_optional_refresh_current2_20260522`; `scratch/sm120_rounds/current-sm120-audit.{json,md}` passes `132` checks with `43` trainer rows, `9` project Torch-fastest rows, and `0` active promotions.
- Cleanup: no `model_*.bin` or `state_*.bin` checkpoint payloads remain under `log124M`.
- Decision: promoted and published as the current SM120 native selection round. The post-audit wrapper proof is faster than `new-goal.md`, the previous promoted direct proof, and the user's pasted `2462.236 ms` first-five sample. The current audit has no active promotion candidates from the captured benchmark matrix.

## Promotion Rule

A combination can replace the default trainer only if it passes correctness,
records the full round manifest, and improves end-to-end training over the
current stable x10 baseline. A three-step smoke can shortlist a combination,
but promotion requires a longer x10 stability run.
