# batch_shape_sweep_20260522

Purpose: test whether the reported `estimated maximum batch size: 73` can improve TinyStories training throughput by reducing or reshaping gradient accumulation, without changing the selected kernel stack.

Selected backend mix for all runs: cuBLASLt-backed trainer route, packed-QKV TK attention, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar AdamW grad scale, ZeRO stage 1, BF16, `gelu_fusion = 1`.

## `b73_ga7_x5`

- Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_b73_ga7_x5_20260522 -v 250 -s 0 -g 144 -h 0 -b 73 -t 1024 -d 523264 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 0 -y 0 -e d12 -x 5`
- Microbatch/accumulation: `B=73`, `grad_accum_steps=7`, `total_batch_size=523264`.
- Memory: activations `29092 MiB`; device usage `32397 MiB / 32606 MiB`.
- Step timings: 2823.95, 2832.62, 2835.00, 2844.93, 2868.52 ms.
- Visible x5 average: 2841.004000 ms.
- Visible first-three average: 2830.523333 ms.
- Normalized throughput from visible average: 184182.8 tok/s.
- Losses: initial val loss `11.033133`, final val loss `10.188684`.
- Decision: rejected. The near-maximum resident microbatch is memory-saturated and about `12.2%` lower throughput than the latest direct selected-stack rerun.

## `b70_ga7_x3`

- Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_b70_ga7_x3_20260522 -v 250 -s 0 -g 144 -h 0 -b 70 -t 1024 -d 501760 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 0 -y 0 -e d12 -x 3`
- Microbatch/accumulation: `B=70`, `grad_accum_steps=7`, `total_batch_size=501760`.
- Memory: activations `27899 MiB`; device usage `31203 MiB / 32606 MiB`.
- Step timings: 2395.42, 2382.68, 2393.00 ms.
- Visible first-three average: 2390.366667 ms.
- Normalized throughput from visible average: 209909.2 tok/s.
- Losses: initial val loss `11.033130`, final val loss `10.610133`.
- Decision: rejected as a training-default change. The step time is lower only because the run processes fewer tokens per optimizer step. Normalized throughput is effectively tied with the latest selected-stack rerun, not a significant speed improvement.

## `b65_ga8_x3`

- Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_b65_ga8_x3_20260522 -v 250 -s 0 -g 144 -h 0 -b 65 -t 1024 -d 532480 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 0 -y 0 -e d12 -x 3`
- Microbatch/accumulation: `B=65`, `grad_accum_steps=8`, `total_batch_size=532480`.
- Memory: activations `25911 MiB`; device usage `29215 MiB / 32606 MiB`.
- Step timings: 2547.52, 2541.77, 2547.06 ms.
- Visible first-three average: 2545.450000 ms.
- Normalized throughput from visible average: 209188.9 tok/s.
- Losses: initial val loss `11.033172`, final val loss `10.609865`.
- Decision: rejected. Keeping the same accumulation count with a slightly larger microbatch reduces normalized throughput versus the selected `B=64, grad_accum_steps=8` script.

## `b71_ga7_x3`

- Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_b71_ga7_x3_20260522 -v 250 -s 0 -g 144 -h 0 -b 71 -t 1024 -d 508928 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 0 -y 0 -e d12 -x 3`
- Microbatch/accumulation: `B=71`, `grad_accum_steps=7`, `total_batch_size=508928`.
- Memory: activations `28297 MiB`; device usage `31601 MiB / 32606 MiB`.
- Step timings: 2423.41, 2417.18, 2421.05 ms.
- Visible first-three average: 2420.546667 ms.
- Normalized throughput from visible average: 210252.5 tok/s.
- Losses: initial val loss `11.033285`, final val loss `10.610292`.
- Decision: x10-gated because the x3 normalized-throughput edge was small but positive.

## `b72_ga7_x3`

- Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_b72_ga7_x3_20260522 -v 250 -s 0 -g 144 -h 0 -b 72 -t 1024 -d 516096 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 0 -y 0 -e d12 -x 3`
- Microbatch/accumulation: `B=72`, `grad_accum_steps=7`, `total_batch_size=516096`.
- Memory: activations `28694 MiB`; device usage `31999 MiB / 32606 MiB`.
- Step timings: 2457.74, 2453.53, 2458.33 ms.
- Visible first-three average: 2456.533333 ms.
- Normalized throughput from visible average: 210089.5 tok/s.
- Losses: initial val loss `11.033197`, final val loss `10.610055`.
- Decision: rejected. It is slower than `B=71/GA=7` and still below the threshold for a significant training-default change.

## `b71_ga7_x10`

- Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_b71_ga7_x10_20260522 -v 250 -s 0 -g 144 -h 0 -b 71 -t 1024 -d 508928 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 0 -y 0 -e d12 -x 10`
- Microbatch/accumulation: `B=71`, `grad_accum_steps=7`, `total_batch_size=508928`.
- Memory: activations `28297 MiB`; device usage `31601 MiB / 32606 MiB`.
- Step timings: 2421.76, 2415.94, 2422.27, 2434.19, 2435.77, 2438.00, 2431.10, 2439.64, 2443.31, 2443.45 ms.
- Trainer average: 2433.741358 ms.
- Visible x10 average: 2432.543000 ms.
- Visible first-three average: 2419.990000 ms.
- Visible first-five average: 2425.986000 ms.
- Normalized throughput from visible x10 average: 209216.4 tok/s.
- Normalized throughput from trainer average: 209113.4 tok/s.
- Losses: initial val loss `11.033285`, final val loss `9.483779`.
- Decision: rejected. The longer gate failed the small x3 signal and normalized throughput is lower than the latest selected `B=64/GA=8` direct rerun.

## `b68_ga8_x3`

- Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_b68_ga8_x3_20260522 -v 250 -s 0 -g 144 -h 0 -b 68 -t 1024 -d 557056 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 0 -y 0 -e d12 -x 3`
- Microbatch/accumulation: `B=68`, `grad_accum_steps=8`, `total_batch_size=557056`.
- Memory: activations `27104 MiB`; device usage `30409 MiB / 32606 MiB`.
- Step timings: 2651.27, 2645.59, 2649.39 ms.
- Visible first-three average: 2648.750000 ms.
- Normalized throughput from visible average: 210309.0 tok/s.
- Losses: initial val loss `11.033145`, final val loss `10.609665`.
- Decision: x10-gated because the x3 normalized-throughput edge was small but positive while keeping `grad_accum_steps=8`.

## `b68_ga8_x10`

- Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_b68_ga8_x10_20260522 -v 250 -s 0 -g 144 -h 0 -b 68 -t 1024 -d 557056 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 0 -y 0 -e d12 -x 10`
- Microbatch/accumulation: `B=68`, `grad_accum_steps=8`, `total_batch_size=557056`.
- Memory: activations `27104 MiB`; device usage `30409 MiB / 32606 MiB`.
- Step timings: 2647.59, 2643.03, 2648.98, 2653.97, 2664.64, 2665.71, 2669.43, 2670.59, 2672.21, 2668.85 ms.
- Trainer average: 2661.934800 ms.
- Visible x10 average: 2660.500000 ms.
- Visible first-three average: 2646.533333 ms.
- Visible first-five average: 2651.642000 ms.
- Normalized throughput from visible x10 average: 209380.2 tok/s.
- Normalized throughput from trainer average: 209267.3 tok/s.
- Losses: initial val loss `11.033145`, final val loss `9.483319`.
- Decision: rejected. The longer gate failed the small x3 signal and normalized throughput is lower than the latest selected `B=64/GA=8` direct rerun.

## Reference

- Latest direct selected-stack rerun: `direct_train_sm120_cef_user_rerun3_x10_20260522`, `B=64`, `grad_accum_steps=8`, `total_batch_size=524288`, visible x10 average `2498.807000 ms`, normalized throughput `209815.3 tok/s`, visible first-three `2489.346667 ms`.
