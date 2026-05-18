# Changelog

Append-only history of meaningful changes to llm.kittens. Roughly grouped by
milestone. Adds within a milestone are listed in chronological order.

The canonical "what is done / what is left" is [`goal.md`](goal.md). The
changelog is the diary; `goal.md` is the plan.

## 2026-05-17 — SM120 RTX 5090 GEMM fallback and pure-TK tuning

- Added an SM120 cuBLASLt GEMM fallback path for GPT-2 matmul forward,
  backward dInput/dWeight, and fused backward-GELU where available. The
  fallback is the default for `DEVICE_ARCH=SM120`; pure TK remains available
  with `SM120_USE_CUBLASLT_GEMM=0`.
- Added `bench_sm120_matmul`, a focused RTX 5090 benchmark comparing pure TK
  GEMM shapes against the SM120 cuBLASLt fallback for GPT-2 qkv, attention
  projection, MLP, LM-head forward, dInput, and dWeight paths.
- Tuned the pure SM120 TK path with fused bias and bias+GELU epilogues,
  separate K=64 backward GEMM traits, concurrent split-K dWeight, and
  shape-aware qkv split-K. Added explicit dInput and optional fused dGELU
  smoke coverage; this exposed the 8-warp wide NN tile as incorrect for plain
  dInput, so plain dInput now stays on the correct 128x64 path while the
  experimental pure dGELU epilogue remains disabled by default.
- Swept pure-TK dWeight split-K values on RTX 5090. QKV still prefers 16-way
  split-K and most non-QKV shapes prefer the existing 8-way cap. A 4-way square
  dWeight split was faster in the benchmark but failed the dWeight smoke
  tolerance, so it is not enabled.
- Tuned the existing CUDA bias-gradient reduction launch for SM120. RTX 5090
  prefers a 512-thread block for this reduction over the old H100-derived
  768-thread choice; `LLMK_SM120_BIAS_BLOCK_SIZE` remains available for
  follow-up sweeps.
- Tuned SM120 TK attention tile sizes independently for forward and backward.
  The default is now a 32-row forward tile and a 16-row backward tile; full
  32-row attention was faster than the old 16-row default, but 64-row tiles
  were correct and much slower from register pressure. CUDA fallback attention
  backward was pathologically slow for the 3-step GPT-2 shape and was not kept.
- Added an SM120 GPT-2 packed-QKV attention fast path. The trainer now stores
  the QKV projection directly in the saved `(B, T, 3, NH, HS)` activation slot,
  SM120 TK attention loads Q/K/V from that packed layout, and SM120 TK backward
  writes packed dQ/dK/dV directly to the QKV input-gradient buffer. This removes
  the forward QKV permute, forward attention unpermute, and final backward
  QKV-gradient permute for the default RTX 5090 path; fallback and atomic-dQ
  builds keep the original permuted layout.
- Validation on RTX 5090 with the TinyStories 3-step command plus `-x 3`:
  default SM120 cuBLASLt fallback averaged `2662.30 ms`; corrected pure TK
  averaged `3729.76 ms`; the optimized attention split averaged `2630.40 ms`
  best / `2633.44 ms` on a clean rerun; direct attention output plus direct
  backward gradients averaged `2580.96 ms`; the packed-QKV attention path
  averaged `2547.84 ms` with steps `2544.99`, `2544.00`, and `2551.67 ms`;
  and the SM120 512-thread bias-gradient reduction averaged `2547.22 ms`
  steady-state with steps `2569.93`, `2543.22`, and `2551.23 ms`.
  `test_matmul` and `test_attention` pass, including the SM120 packed-QKV
  attention smoke case. cuBLASLt plan caching
  (`2663.15 ms`, `2731.19 ms` with the attention split), heuristic index
  selection (`2687.10 ms` / `2744.01 ms`), min/max-wave selection
  (`2691.49 ms` / `3495.77 ms`), 256 MiB workspace (`2635.88 ms`), and
  8-warp dprep (`2630.79 ms`) did not beat the default fallback plus the
  optimized attention split. `-maxrregcount=128` (`2815.44 ms`) and a
  vectorized dprep helper (`2638.18 ms`) were also slower and are not enabled.
- Extended `bench_sm120_matmul` to include the real MLP up-projection
  bias+GELU/pre-GELU forward path, and later made it report both the fused
  TK epilogue and the pure-TK trainer default of matmul plus explicit CUDA
  GELU. A current pure-TK no-extra-flags rebaseline with the same TinyStories
  3-step command averaged `5028.57 ms`, so it remains far behind the
  cuBLASLt-backed SM120 default. Follow-up pure-TK candidates were rejected:
  `LLMK_SM120_K_TILE=64` averaged `4946.25 ms`, and
  `LLMK_SM120_DWEIGHT_SPLIT_K=32` averaged `4864.60 ms`. The restored default
  SM120 cuBLASLt-backed build averaged `2548.46 ms` with steps `2563.11`,
  `2545.60`, and `2551.31 ms`.
- Changed the automatic GPT-2 `-ge` default so SM120 cuBLASLt builds keep
  fused GELU (`gelu_fusion=1`) while pure SM120 TK builds default to explicit
  CUDA GELU (`gelu_fusion=0`). With no extra macros, the patched pure-TK
  3-step run averaged `3578.03 ms`, versus `5028.57 ms` for the previous
  pure-TK default. `LLMK_SM120_FORCE_DEFAULT_TILE` failed the matmul smoke
  test, and `LLMK_SM120_DWEIGHT_SPLIT_K=32` did not improve the corrected
  pure-TK setup (`3571.30 ms`).
- Verified the SM120 cuBLASLt default should keep fused GELU: the same
  restored default binary with `-ge 0` averaged `2619.29 ms`, slower than the
  fused default.
- Additional pure-TK SM120 sweeps were rejected after benchmark plus smoke or
  3-step validation. Forward-only huge-N thresholds `768` and `2048` did not
  beat the default threshold mix. dWeight `SUPER_M` values `1`, `2`, and `8`
  did not close the dWeight gap, and disabling concurrent split-K streams was a
  large regression. GELU backward blocks `512` and `64` passed `test_gelu` but
  regressed 3-step pure-TK training to `4716.63 ms` and `5342.83 ms`.
  Disabling the SM120 forward wide tile looked fast in the benchmark but failed
  `test_matmul` on the GPT-2 MLP-up and bias+GELU forward cases; huge-N
  `K_TILE=32` nearly matched cuBLASLt for LM-head forward but failed the
  LM-head smoke, `K_TILE=96` was slower, and `K_TILE=128` exceeded SM120 shared
  memory. `LLMK_SM120_GRAD_K_TILE=32` passed `test_matmul` but regressed
  3-step pure-TK training to `4847.39 ms`; a dInput-only split of that idea
  also passed `test_matmul` but regressed to `5350.23 ms`, so it was not kept.
- Final restored SM120 default validation on RTX 5090: `test_matmul` passed
  `8/8`, `test_attention` passed all three smoke shapes including packed-QKV,
  and the TinyStories command with `-x 3` averaged `2536.44 ms` with steps
  `2534.52`, `2533.96`, and `2538.93 ms` (`206.7k`-`206.9k` tok/s). The final
  `bench_sm120_matmul` pass still shows pure TK behind cuBLASLt on every safe
  GPT-2 GEMM shape; the largest remaining gap is attention-projection dWeight
  at `1.74x` slower.
- Additional rejected SM120 pure-TK follow-ups: stricter `cp.async.wait_group 1`
  passed the smoke test but slowed every benchmarked GEMM shape; launch-bounds
  `minBlocks=2` made some direct forward timings look competitive but failed
  the pure-TK MLP-up and bias+GELU smoke cases, while NN-only launch bounds
  passed smoke and regressed 3-step training to `4994.76 ms`. A 512-thread
  split-K dWeight reduction passed smoke but regressed training to `5115.35 ms`,
  pure `-ge 1` regressed to `4897.44 ms`, and shared-memory carveout hints
  worsened the benchmark. Huge-N `K_TILE=32` plus global `wait_group 1` was
  correct but regressed training to `5442.61 ms`; restricting the wait change
  to huge-N failed the pure LM-head smoke. `SUPER_M` values `4` and `16`,
  `DINP_SUPER_M` values `2` and `4`, and re-enabling the wide plain-dInput tile
  also failed to close the cuBLASLt gap. The ThunderKittens B200 BF16 GEMM
  source is not a portable SM120 replacement: ptxas rejects its `tcgen05` and
  `.cta_group::2` instructions for `sm_120a`.
- After removing those rejected hooks, restored the default SM120 cuBLASLt build
  and revalidated on RTX 5090: `test_matmul` passed `8/8`, `test_attention`
  passed all three smoke shapes, `bench_sm120_matmul` still showed pure TK
  behind cuBLASLt on every GPT-2 GEMM row, and the TinyStories command capped
  with `-x 3` averaged `2556.74 ms` with steps `2598.45`, `2548.75`, and
  `2564.74 ms` (`201.8k`-`205.7k` tok/s).
- Rejected an SM120 pure-TK dWeight atomic split-K experiment. Direct BF16
  `atomicAdd` compiled and the all-shape and square-only variants passed
  `test_matmul`, but `bench_sm120_matmul` showed worse or mixed dWeight ratios
  than the restored scratch-plus-reduce path, so the atomic launcher and wrapper
  were removed.
- Rejected two more dWeight-focused TK candidates: `LLMK_SM120_GRAD_K_TILE=128`
  exceeded SM120 shared-memory limits in the wide backward kernels, and a
  128x128 dWeight tile passed `test_matmul` but made every benchmarked dWeight
  row slower than the restored 256x64 path.
- Rejected full K-loop unrolling in the SM120 TK GEMM kernels. The unrolled
  build passed `test_matmul`, but benchmark timings were mixed and all rows
  still trailed cuBLASLt, with most forward and dInput absolute timings worse
  than the restored default.
- Rejected a compile-time-specialized split-K partial reducer for the common
  8- and 16-part dWeight cases. It passed `test_matmul`, but the focused
  benchmark only slightly helped qkv dWeight while regressing several other
  dWeight rows, including attention projection and LM-head.
- Final restored-default cleanup validation on RTX 5090: `test_matmul` passed
  `8/8`, `test_attention` passed all three smoke shapes including packed-QKV,
  and the TinyStories command capped with `-x 3` averaged `2525.99 ms` with
  steps `2529.42`, `2529.39`, and `2522.60 ms` (`207.3k`-`207.6k` tok/s).
  The final `bench_sm120_matmul` pass still showed direct pure-TK GEMM behind
  cuBLASLt on every GPT-2 row; the worst ratio was attention-projection dWeight
  at `1.77x` slower.
- Rejected a TK dWeight in-place register-layout-swap variant. A global version
  passed `test_matmul` but was mixed in the direct benchmark, and a guarded
  qkv/attention/MLP-up dWeight-only version still regressed pure-TK TinyStories
  3-step training to `5904.20 ms`, so the hook was removed.
- Rebaselined the restored pure-TK SM120 path after that rejection:
  `test_matmul` passed `7/7` in the no-cuBLASLt build, default explicit GELU
  averaged `5743.60 ms`, and `-ge 1` averaged `5416.18 ms`. The GPT-2 trainer
  now defaults SM120 builds to `gelu_fusion=1`; this improves the current
  pure-TK path but still does not close the cuBLASLt gap.
- Rejected enabling the pure-TK SM120 fused dGELU epilogue by default. It
  passes `test_matmul` (`8/8`) with `LLMK_SM120_FUSE_DGELU=1`, but default-source
  TinyStories 3-step reruns averaged `5412.11 ms` and `5358.96 ms`, while an
  apples-to-apples disabled-dGELU build averaged `5056.27 ms`. The source
  default was restored to disabled, passed `test_matmul` (`7/7`), and averaged
  `5236.04 ms` on a final pure-TK 3-step run. The hook remains available for
  A/B testing and disabled by default.
- Restored the default cuBLASLt-backed SM120 build after the fused-dGELU A/B:
  `test_matmul` passed `8/8`, `test_attention` passed all three smoke shapes,
  `bench_sm120_matmul` still showed pure TK behind cuBLASLt on every GPT-2 GEMM
  row, and the TinyStories command capped with `-x 3` averaged `2540.58 ms`
  before the extra rejected tuning sweeps and `2544.26 ms` after the final
  rebuild/revert.
- Rejected additional SM120 GEMM tuning variants after direct benchmarks or
  smoke tests: lowering `LLMK_SM120_HUGE_N_THRESHOLD` to `768` regressed
  dInput/dWeight heavily, `LLMK_SM120_DWEIGHT_SUPER_M=16` and `=3` did not
  improve the dWeight rows enough to offset regressions, and raising
  `__launch_bounds__` occupancy for NT forward made the MLP-up and fused
  bias+GELU smoke cases numerically incorrect.
- Rejected a dWeight-only force-default-tile variant. The correctly scoped
  `LLMK_SM120_DWEIGHT_FORCE_DEFAULT_TILE` A/B passed `test_matmul` and improved
  some direct dWeight benchmark rows, but pure-TK TinyStories 3-step training
  regressed to `5671.51 ms`, so the default 256x64 TN tile remains in place.
- Rejected a 128x192 huge-N tile for the LM-head path. It passed `test_matmul`
  and improved direct LM-head forward timing, but pure-TK TinyStories 3-step
  training regressed to `5995.44 ms`; the huge-N aliases remain on 128x128.
- Rebuilt the restored default cuBLASLt-backed SM120 path after those rejected
  experiments: `test_matmul` passed `8/8`, `test_attention` passed all three
  smoke shapes, the focused GEMM benchmark still had pure TK behind every
  cuBLASLt row, and the TinyStories command capped with `-x 3` averaged
  `2538.98 ms`.
- Rejected allowing 16-way split-K for every dWeight shape. The A/B passed
  `test_matmul`, but the direct benchmark was mixed and pure-TK TinyStories
  3-step training regressed to `6643.29 ms`; non-QKV dWeight remains capped at
  8-way split-K.
- Rejected overlapping pure-TK backward dInput and dWeight GEMMs with a side
  stream. The event-ordered variant passed `test_matmul`, but TinyStories
  3-step training regressed to `6439.10 ms`, so `matmul_backward` stays
  serial within the caller stream.
- Rebuilt the restored default cuBLASLt-backed SM120 path after the split-K and
  overlap rejections: `test_matmul` passed `8/8`, `test_attention` passed all
  three smoke shapes, the focused GEMM benchmark still showed every pure-TK row
  behind cuBLASLt, and the TinyStories command capped with `-x 3` averaged
  `2594.86 ms`.
- Rejected disabling dWeight split-K with `LLMK_SM120_DWEIGHT_SPLIT_K=1`. The
  smoke test passed, but the focused benchmark made dWeight rows much slower,
  including attention-projection dWeight at `6.16x` slower than cuBLASLt.
- Rejected a dWeight-only `K_TILE=128` experiment for SM120. The scoped build
  passed `test_matmul`, but `bench_sm120_matmul` made every dWeight row slower
  than the restored default, including LM-head dWeight at `1.70x` slower than
  cuBLASLt. The temporary dWeight K-tile hook was removed.
- Added a `train_gpt2cu` source guard so SM120 training cannot be built with
  TK attention disabled. The standalone attention smoke can still exercise the
  CUDA fallback path, but the trainer now refuses that fallback configuration
  instead of entering the path that spun outside the intended GPU kernel route.
- Rebuilt the restored default SM120 path with cuBLASLt GEMMs and TK attention
  after the guard: `test_matmul` passed `8/8`, `test_attention` passed all
  three smoke shapes, `bench_sm120_matmul` still showed pure-TK GEMM rows
  behind cuBLASLt, and the TinyStories command capped with `-x 3` averaged
  `2565.60 ms`. An explicit
  `EXTRA_NVCC_FLAGS='-DLLMK_DISABLE_TK_MHA'` trainer build now fails at the
  source guard as intended.
- Rejected another SM120 pure-TK GEMM sweep after focused benchmark runs:
  a compact 64x64 forward wide tile passed `test_matmul` and helped
  attention/fcproj forward, but left dInput/dWeight behind cuBLASLt; shared
  carveout hints were mixed or slower; direct column loads for dInput and
  dWeight were mixed, including A-only, B-only, and non-square-gated dWeight
  forms that still left the best dWeight rows at least `1.07x` slower than
  cuBLASLt and attention-projection dWeight above `1.5x`. A probe that tried
  to compile the existing Hopper WGMMA GEMM header for SM120 failed because the
  SM120 TK warpgroup surface does not expose the Hopper
  `warpgroup::mma_async_wait` API. The rejected hooks and probe file were
  removed. After cleanup, the default SM120 cuBLASLt-backed build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), the
  focused benchmark still showed pure-TK GEMM behind cuBLASLt with
  attention-projection dWeight at `1.59x`, and the TinyStories command capped
  with `-x 3` averaged `2619.90 ms` across the three step lines
  (`2572.54`, `2652.46`, `2634.70 ms`; trainer-reported total average
  `2643.58 ms`).
- Rejected a single-launch split-K dWeight prototype that mapped split parts to
  `grid.y` instead of launching one TN kernel per part on side streams. It
  passed `test_matmul`, but the focused benchmark still left every dWeight row
  slower than cuBLASLt; requested split counts `4`, default `16`/`8`, and `32`
  were all behind. A second Hopper-wrapper probe forced the `KITTENS_SM90` API
  while compiling for `sm_120a`; ptxas rejected `wgmma.fence`,
  `wgmma.mma_async`, `wgmma.commit_group`, and `wgmma.wait_group`, confirming
  the H100 WGMMA path is not portable to SM120. The temporary split-K launcher
  and probe file were removed.
- Pruned SM120 trainer-side synchronizations. Forward no longer hard-syncs at
  the end, training/validation input and target copies are stream-ordered,
  residual/loss/optimizer memsets are asynchronous, non-final micro-steps no
  longer device-synchronize after backward, and the update path now relies on
  the training-loop end event for timing synchronization. The final restored
  3-step cuBLASLt-backed rerun averaged `2550.94 ms`, with the same TinyStories
  loss/norm trace as the earlier validated runs. An async grad-norm scalar copy
  was tested and rejected after it regressed the 3-step average to
  `2586.19 ms`.
- Added a pure SM120 TK 128x96 forward tile for GPT-2 projection widths
  divisible by 96 (`LLMK_SM120_FORWARD_N96=1` by default). The pure-TK matmul
  smoke passed `7/7`, and the TinyStories 3-step pure-TK run improved to
  `2956.35 ms` with steps `2938.72`, `2955.41`, and `2957.30 ms`. It is still
  slower than the cuBLASLt-backed SM120 default and the user's llm.c baseline,
  so the goal remains open for the remaining GEMM gaps.
- Re-tested pure SM120 TK fused dGELU after the N96 forward tile. The
  `LLMK_SM120_FUSE_DGELU=1` build passed `test_matmul` (`8/8`) and improved
  the 3-step pure-TK average to `2940.72 ms`, so fused dGELU is now enabled by
  default for pure SM120 builds when the trainer uses `-ge 1`.
- Lowered the pure SM120 dWeight split-K default from 16 to 8 after the current
  N96+dGELU trainer favored fewer qkv part launches. The split-K=8 A/B passed
  `test_matmul` (`8/8`) and averaged `2936.47 ms` on its first 3-step
  TinyStories run; the no-extra-flags source rerun averaged `2928.99 ms` with
  steps `2926.24`, `2925.07`, and `2932.91 ms`. Rejected follow-ups:
  96-column backward tiles made every direct dInput and dWeight benchmark row
  worse; forcing the LM-head through N96 by raising
  `LLMK_SM120_HUGE_N_THRESHOLD` failed the MLP-up and LM-head smoke rows;
  split-K=4 failed smoke; split-K=32, `LLMK_SM120_DWEIGHT_SUPER_M=6`,
  `LLMK_SM120_K_TILE=64`, and attention backward block 32 were correct but
  slower or mixed; `LLMK_SM120_DINP_SUPER_M=16` and
  `LLMK_SM120_HUGE_N_K_TILE=32` failed smoke.
- Restored and revalidated the normal cuBLASLt-backed SM120 build after the
  pure-TK changes. `test_matmul` passed `8/8`, `test_attention` passed all
  three smoke shapes, `bench_sm120_matmul` still showed pure TK behind cuBLASLt
  on the remaining GEMM rows, and the TinyStories command capped with `-x 3`
  averaged `2540.53 ms` with steps `2548.52`, `2538.77`, and `2542.29 ms`.
- Continued pure SM120 TK tuning on RTX 5090. In-place register-layout swaps,
  the cosh-free fast dGELU derivative, the dGELU-only approximate tanh path
  (`LLMK_SM120_APPROX_DGELU_TANH=1`), and `LLMK_SM120_DWEIGHT_SUPER_M=2` are
  now the source defaults after `test_matmul` passed `8/8` and the best
  TinyStories 3-step pure-TK run improved to `2845.57 ms`
  (`log124M/5090_puretk_dwsuperm2_x3`). The source-default rerun averaged
  `2851.49 ms` (`2843.22`, `2850.45`, `2852.54 ms`), still slower than the
  user's llm.c baseline average of `2818.23 ms`.
- Rejected the latest pure-TK follow-ups after smoke, benchmark, compile, or
  3-step validation: approximate tanh in forward GELU failed the MLP-up smoke
  row; the huge-N wide forward route regressed 3-step training to `2891.99 ms`;
  `LLMK_SM120_DWEIGHT_SPLIT_K_STREAMS=0` made dWeight rows much slower;
  `LLMK_SM120_DWEIGHT_SPLIT_K=2` failed smoke and `=16` regressed to
  `2870.05 ms`; `LLMK_SM120_DINP_SUPER_M=4` was mixed and `=12` failed smoke;
  `LLMK_SM120_ATTN_BWD_BLOCK=8` is not a valid kernel tile and
  `LLMK_SM120_ATTN_FWD_BLOCK=16` regressed to `2939.27 ms`;
  `LLMK_SM120_DWEIGHT_SUPER_M=1` failed smoke and `=3` regressed to
  `2874.43 ms`; exact dGELU (`LLMK_SM120_APPROX_DGELU_TANH=0`) was slower than
  the approximate default at `2857.50 ms`. Pure TK therefore remains short of
  the llm.c baseline and still behind cuBLASLt on the focused GEMM benchmark.
- Rebuilt and revalidated the normal cuBLASLt-backed SM120 default after the
  latest pure-TK default changes. `test_matmul` passed `8/8`, `test_attention`
  passed all three smoke shapes including packed-QKV, and
  `bench_sm120_matmul` still showed pure TK behind cuBLASLt on most GEMM rows
  even though attention-projection dInput and fcproj forward were faster than
  cuBLASLt in this run. The TinyStories command capped with `-x 3` averaged
  `2602.73 ms` by the trainer report, with step lines `2579.31`, `2571.45`,
  and `2634.01 ms`; this remains faster than the supplied llm.c baseline but
  does not satisfy the pure-TK kernel-outperformance goal.
- Rejected three more dWeight-focused pure-TK candidates. A direct-first-part
  split-K reducer, which wrote split 0 directly to `dweight` and reduced only
  the remaining scratch partials, passed `test_matmul` but regressed 3-step
  pure-TK training to `5024.85 ms` and shifted the norm trace. Streaming-cache
  loads for split-K partial reduction passed smoke but regressed key benchmark
  rows, including attention-projection and LM-head dWeight. A 256x128 TN
  dWeight tile compiled but made the unrelated GPT-2 MLP-up forward smoke row
  fail reproducibly, so all three hooks were removed.
- Rejected more SM120 pure-TK micro-kernel A/Bs. Disabling the SM120 warp-id
  remap was smoke-correct but worsened or mixed the focused GEMM benchmark, so
  it was removed without a 3-step run. A plain-dInput-only
  `LLMK_SM120_DINP_K_TILE=128` variant compiled after excluding the 128x96
  alias but made every dInput benchmark row worse, with LM-head dInput at
  `1.35x` slower than cuBLASLt. An rsqrt-based dGELU tanh approximation failed
  the fused-dGELU smoke row, and an unclamped rational dGELU tanh approximation
  passed smoke but regressed 3-step pure-TK training to `5299.34 ms` and shifted
  the norm trace. These hooks were removed.
- Rejected additional macro-only SM120 sweeps. `LLMK_SM120_K_TILE=16` failed
  the first square matmul smoke with an illegal memory access. Narrowing the
  huge-N forward K tile to `LLMK_SM120_HUGE_N_K_TILE=48` kept other rows correct
  but failed the GPT-2 LM-head smoke with max diff `10.5781`. Disabling fused
  bias (`LLMK_SM120_FUSE_BIAS=0`) passed smoke but badly regressed forward
  benchmark rows, including qkv and attention projection at `1.48x` and `1.47x`
  slower than cuBLASLt. `LLMK_SM120_SUPER_M=6` failed the plain dInput smoke
  row, so the current defaults remain in place.
- Rejected a dWeight TN N-swizzle experiment before benchmarking. The disabled
  compile-time hook grouped consecutive output-N tiles for each output-M tile
  to try to reuse the A tile in dWeight, but
  `EXTRA_NVCC_FLAGS='-DLLMK_SM120_DWEIGHT_SWIZZLE_N=1 -DLLMK_SM120_DWEIGHT_SUPER_N=8'`
  failed `test_matmul` on the GPT-2 MLP-up forward row with max diff `6.0000`,
  so the hook was removed.
- Rejected a packed BF16-pair split-K partial reducer. The
  `LLMK_SM120_DWEIGHT_REDUCE_BF162=1` build compiled, but `test_matmul` failed
  both dWeight rows at max diff `0.5000` against the strict `< 0.50` tolerance,
  so it was removed without benchmarking.
- Rejected a dInput-only 128-column NN tile route. The
  `LLMK_SM120_DINP_N128=1` build passed `test_matmul` (`8/8`), but
  `bench_sm120_matmul` regressed every dInput row, including LM-head dInput to
  `1.32x` slower than cuBLASLt. The required TinyStories 3-step pure-TK
  validation then averaged `6849.22 ms`, so the hook was removed.
- Rejected a sigmoid-approximate forward GELU hook for the pure SM120 TK path.
  The `LLMK_SM120_FORWARD_GELU_SIGMOID=1` build passed `test_matmul` (`8/8`)
  and improved the direct fused MLP-up benchmark row to `1537.74 us`, but that
  was still slower than the cuBLASLt row at `1479.52 us` and most backward GEMM
  rows remained behind. The required TinyStories 3-step pure-TK validation made
  no step progress and was terminated as CPU-bound after roughly two minutes
  before initial validation loss, with the trainer process at about `99.7%`
  CPU, so the hook was removed.
- Rejected an rsqrt-based approximate forward GELU hook for the pure SM120 TK
  path. The `LLMK_SM120_FORWARD_GELU_RSQRT=1` build passed `test_matmul`
  (`8/8`) and improved the direct fused MLP-up benchmark row to `1521.15 us`,
  but it still trailed cuBLASLt at `1435.28 us`, and the direct dInput/dWeight
  rows stayed behind. The required TinyStories 3-step validation again made no
  step progress after allocation and was terminated with the trainer at about
  `99.6%` CPU, so the hook was removed.
- Rejected a narrower dWeight-only N-swizzle retry. The hook limited the
  alternate N-grouped grid order to `A^T*B` launches and used the normal grid
  for forward/dInput kernels, but the enabled build still failed `test_matmul`
  on the GPT-2 MLP-up forward row with max diff `6.1250`, so it was removed
  before benchmark or training.
- Rejected a macro-only `LLMK_SM120_DWEIGHT_SUPER_M=5` sweep. It passed
  `test_matmul` (`8/8`) and slightly improved some direct dWeight rows, but
  `bench_sm120_matmul` still had all dWeight rows behind cuBLASLt, with
  attention-projection dWeight at `1.68x` slower and attention-projection
  dInput at `1.11x` slower. The required 3-step validation was terminated
  before initial validation output when the trainer sat at about `99.6%` CPU,
  so the source default remains `LLMK_SM120_DWEIGHT_SUPER_M=2`.
- Rejected a hybrid SM120 default-path experiment that routed only GPT-2
  fcproj forward (`N=768,K=3072`) through the TK forward+bias kernel while
  keeping the rest of the cuBLASLt-backed path unchanged. `test_matmul` and
  `test_attention` passed, but the required TinyStories 3-step validation
  regressed to `3230.19 ms` by the trainer report, slower than the restored
  cuBLASLt default and slower than the supplied llm.c baseline, so the hook was
  removed.
- Rejected a cuBLASLt fallback workspace sweep with
  `LLMK_SM120_CUBLASLT_WORKSPACE_MB=512`. `test_matmul` and `test_attention`
  passed, and the focused benchmark did not show a clear win over the restored
  default. The required TinyStories 3-step validation made no progress before
  initial validation output and was terminated with the trainer at about
  `99.6%` CPU, so the workspace default stays at `128` MiB.
- Rejected a dWeight-only K-tile split that kept dInput on
  `LLMK_SM120_GRAD_K_TILE=64` while testing `LLMK_SM120_DWEIGHT_K_TILE=32` for
  TN launches. The build passed `test_matmul` (`8/8`) and improved LM-head
  dWeight in `bench_sm120_matmul` to `1.13x` slower than cuBLASLt, but it
  regressed the other dWeight rows and the required 3-step validation was
  terminated before initial validation output with the trainer at about
  `99.5%` CPU. The temporary dWeight-only trait hook was removed.
- Rejected a CUDA bias-gradient reduction block-size sweep with
  `LLMK_SM120_BIAS_BLOCK_SIZE=1024`. The cuBLASLt-backed TinyStories 3-step
  validation completed with the expected finite loss/norm trace, but regressed
  to `3244.20 ms` by the trainer report, so the source default remains the
  previously validated `512`-thread block.
- Rebaselined the current source-default SM120 cuBLASLt-backed build after the
  rejection-only commits. `test_matmul` passed `8/8`, `test_attention` passed
  all three smoke shapes, and the TinyStories command capped with `-x 3`
  completed with the expected finite loss/norm trace, but averaged
  `3245.60 ms` by the trainer report. The source diff since the prior tuning
  commit was changelog-only, so this rebaseline records current runtime state
  rather than a source regression.
- Re-ran the restored tracked `train_gpt2cu` binary without rebuilding to
  separate build/codegen effects from runtime state. It produced the same
  finite TinyStories trace and averaged `3235.36 ms`, confirming the current
  slowdown is not caused by the latest rebuild flags or rejected source hooks.
- Captured GPU clocks during another current-default 3-step run. The run
  averaged `3239.00 ms`, while `nvidia-smi` samples showed normal boost under
  load (`~2.8 GHz` SM clock, `13.8 GHz` memory clock, `99-100%` utilization,
  and roughly `500-575 W`), so the current slowdown is not a low-clock or
  power-limit artifact. A fresh `-ge 0` retest then averaged `3312.72 ms`,
  confirming the SM120 default should keep GELU fusion enabled even in the
  current runtime state.
- Retested the same cuBLASLt-backed 3-step run with ZeRO disabled via `-z 0`.
  It averaged `3262.44 ms`, slightly slower than the current `-z 1` rebaseline,
  so single-process ZeRO-1 overhead is not the current SM120 bottleneck.
- Changed the SM120 `train_gpt2cu` default to disable FP32 master weights unless
  `-w` is passed explicitly. On RTX 5090, the explicit `-w 0` probe averaged
  `2481.98 ms`, and the patched source-default run of the user's TinyStories
  command capped with `-x 3` showed `use_master_weights disabled` and averaged
  `2481.12 ms` with steps `2484.93`, `2478.22`, and `2484.03 ms`. This restores
  a significant margin over the supplied llm.c baseline while preserving
  explicit `-w 1` for master-weight runs.
- Rebaselined the pure SM120 TK path after the no-master default. The
  `SM120_USE_CUBLASLT_GEMM=0` build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the user's TinyStories command
  capped with `-x 3` still averaged `3335.85 ms` with steps `3343.69`,
  `3330.50`, and `3341.21 ms`. The finite loss/norm trace confirms the path is
  stable, but it remains behind both the supplied llm.c baseline and the
  cuBLASLt-backed SM120 default, so the pure-TK kernel-outperformance goal is
  still open.
- Rejected a macro-wide `LLMK_SM120_GRAD_K_TILE=48` sweep. It compiled, but
  `test_matmul` aborted before training because the shared gradient tile also
  applies to dInput shapes whose reduction dimension is not divisible by `48`.
  No 3-step TinyStories validation was run for this invalid build.
- Rejected a dWeight-scoped `LLMK_SM120_DWEIGHT_K_TILE=48` hook. The temporary
  hook kept dInput on the valid `64`-wide gradient tile, but `test_matmul`
  still aborted because the dWeight reduction dimension is not divisible by
  `48` for the covered shapes. The hook was removed and no 3-step TinyStories
  validation was run.
- Rejected `LLMK_SM120_GRAD_K_TILE=16`. It compiled, but `test_matmul` failed
  the GPT-2 MLP-up and LM-head forward smoke rows and then hit an illegal memory
  access in the fused dGELU dInput smoke. No benchmark or 3-step TinyStories
  validation was run for this incorrect build.
- Rejected `LLMK_SM120_BACKWARD_N96=1`. It passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark made the
  backward rows worse (`dInp` roughly `1.29x-1.38x` and dWeight roughly
  `1.32x-1.81x` slower than cuBLASLt). The required TinyStories 3-step
  validation completed with a finite trace but regressed to `3681.43 ms`
  average with steps `3659.10`, `3652.39`, and `3710.48 ms`, so backward N96
  remains disabled.
- Rejected `LLMK_SM120_INPLACE_LAYOUT_SWAP=0`. It compiled, but `test_matmul`
  failed the fused dGELU dInput row (`8.1250` max diff versus `0.50`
  tolerance), so the explicit-layout-swap variant was not benchmarked or
  validated with a 3-step TinyStories run.
- Rejected `LLMK_SM120_FAST_DGELU=0`. It passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark still
  left all material pure-TK backward rows behind cuBLASLt and the required
  TinyStories 3-step validation averaged `3340.32 ms` with steps `3341.56`,
  `3334.87`, and `3345.78 ms`. That is slightly slower than the pure-TK
  no-master rebaseline, so the cosh-free dGELU derivative remains enabled.
- Rejected a current no-master retest of `LLMK_SM120_HUGE_N_FORWARD_WIDE=1`.
  It passed `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes)
  and improved the direct LM-head forward row to `1.20x` slower than cuBLASLt,
  but dInput/dWeight stayed behind and the required TinyStories 3-step
  validation regressed to `3541.71 ms` average with steps `3553.81`, `3531.25`,
  and `3552.17 ms`.
- Rejected the current pure-TK explicit-GELU runtime path (`-ge 0`) under the
  no-master default. The current pure-TK build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the user's TinyStories command
  capped with `-x 3 -ge 0` regressed to `3659.73 ms` average with steps
  `3634.56`, `3652.68`, and `3666.78 ms`, so fused GELU remains the SM120
  default.
- Rejected a temporary huge-N `256x128` tile selector for the pure SM120 TK
  LM-head forward path. It passed `test_matmul` (`8/8`) and `test_attention`
  (all three smoke shapes) and improved direct LM-head forward to `24571.88 us`
  (`1.12x` slower than cuBLASLt), but the required TinyStories 3-step
  validation still regressed to `3545.80 ms` average with steps `3507.21`,
  `3532.32`, and `3559.29 ms`. The selector hook was removed.
- Rejected a temporary huge-N `64x128` tile shape. It compiled, but
  `test_matmul` failed the GPT-2 MLP-up and LM-head rows (`7.0625` and
  `10.3359` max diff versus `0.50` tolerance), so no benchmark or TinyStories
  3-step validation was run and the source edit was reverted.
- Rejected explicit master weights for the current pure SM120 TK path (`-w 1`).
  The pure-TK build passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), but the required TinyStories validation stalled before the
  first step and was killed at `99.7%` CPU after roughly `01:33`, matching the
  known invalid CPU-bound failure mode.
- Re-tested SM120 cuBLASLt plan caching under the no-master default with
  `LLMK_SM120_CACHE_CUBLASLT_PLANS`. It passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), and the required TinyStories
  3-step validation averaged `2480.25 ms` with steps `2485.15`, `2477.94`, and
  `2482.55 ms`. This is only a noise-level improvement over the accepted
  no-master default, so plan caching remains opt-in.
- Re-tested pure SM120 TK `LLMK_SM120_DWEIGHT_SPLIT_K=16` under the no-master
  default. It passed `test_matmul` (`8/8`) and `test_attention` (all three smoke
  shapes), and improved the current 3-step pure-TK average only slightly to
  `3329.02 ms` with steps `3338.76`, `3328.02`, and `3330.01 ms`. The focused
  benchmark remained mixed and all key pure-TK dWeight rows still trailed
  cuBLASLt, so split-K stays at the source default of `8`.
- Rejected pure SM120 TK `LLMK_SM120_DWEIGHT_SUPER_M=7`. It passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but the
  focused benchmark remained behind cuBLASLt on all dWeight rows and the
  required TinyStories 3-step validation averaged `3337.92 ms` with steps
  `3353.31`, `3333.54`, and `3342.30 ms`, slightly slower than the current
  pure-TK no-master rebaseline.
- Re-tested the dWeight-only `LLMK_SM120_DWEIGHT_K_TILE=32` split under the
  current no-master pure-TK path. The temporary trait hook kept dInput on the
  valid `LLMK_SM120_GRAD_K_TILE=64` route and passed `test_matmul` (`8/8`) plus
  `test_attention` (all three smoke shapes), but the focused benchmark still
  left every dWeight row behind cuBLASLt (`1.56x`, `1.79x`, `1.40x`, `1.48x`,
  and `1.11x` slower for qkv, attention projection, MLP-up, MLP projection,
  and LM-head). The required TinyStories 3-step validation regressed badly to
  `5272.44 ms` with steps `5036.14`, `5312.39`, and `5232.49 ms`, so the hook
  was removed again.
- Rejected a macro-only `LLMK_SM120_DWEIGHT_SUPER_M=4` retest under the
  no-master pure-TK path. It passed `test_matmul` (`8/8`) and `test_attention`
  (all three smoke shapes), but the focused benchmark still had all dWeight
  rows behind cuBLASLt (`1.48x`, `1.57x`, `1.31x`, `1.32x`, and `1.23x`
  slower), and the required TinyStories 3-step validation regressed to
  `3618.29 ms` with steps `3660.48`, `3614.09`, and `3622.49 ms`.
- Rejected `LLMK_SM120_DWEIGHT_SPLIT_K=64`. The build passed `test_matmul`
  (`8/8`) and `test_attention` (all three smoke shapes), but the focused
  benchmark showed the extra qkv split parallelism still left qkv dWeight
  `1.47x` slower than cuBLASLt and worsened LM-head dWeight to `1.24x` slower.
  The required TinyStories 3-step validation averaged `3614.52 ms` with steps
  `3630.85`, `3624.55`, and `3604.48 ms`, so the default split-K cap remains
  unchanged.
- Rejected a temporary dWeight-only 64x64 TN tile for the wide dWeight route.
  The hook passed `test_matmul` (`8/8`) and `test_attention` (all three smoke
  shapes) and improved the direct qkv dWeight row to `1.34x` slower than
  cuBLASLt, but the other dWeight rows stayed between `1.35x` and `1.63x`
  slower and the required TinyStories 3-step validation averaged
  `3624.98 ms` with steps `3627.30`, `3616.67`, and `3633.29 ms`. The hook was
  removed and the wide dWeight route remains on the current 256x64 tile.
- Rejected a temporary dWeight-only 128x32 TN tile for the wide dWeight route
  before benchmarking. The build compiled, but `test_matmul` failed the GPT-2
  MLP-up forward smoke row with max diff `5.5156` versus the `0.50` tolerance,
  so no focused benchmark or TinyStories 3-step validation was run and the hook
  was removed.
- Rejected promoting `LLMK_SM120_DWEIGHT_SPLIT_K=16` to the source default
  under the current no-master pure-TK runtime. The source-default candidate
  passed `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes),
  but the focused benchmark still left every dWeight row behind cuBLASLt and
  the required TinyStories 3-step validation regressed to `3616.37 ms` with
  steps `3575.14`, `3607.16`, and `3625.58 ms`. The wrapper default remains
  `8`, with the qkv-only 16-way split left as an explicit A/B macro.
- Rebaselined the clean current-source pure SM120 TK path after the rejection
  rounds. The rebuilt `test_matmul` initially had a transient MLP-up forward
  smoke failure but passed `8/8` on the immediate rerun; `test_attention`
  passed all three smoke shapes. The focused benchmark still left the material
  pure-TK GEMM rows behind cuBLASLt except fcproj forward, and the TinyStories
  command capped with `-x 3` averaged `3621.93 ms` with steps `3614.90`,
  `3618.94`, and `3624.92 ms`, so the current pure-TK path is slower than the
  earlier no-master rebaseline and remains behind the supplied llm.c baseline.
- Rebaselined the current cuBLASLt-backed SM120 default under the same
  no-master runtime. `test_matmul` passed `8/8`, `test_attention` passed all
  three smoke shapes, and the TinyStories command capped with `-x 3` averaged
  `2482.82 ms` with steps `2486.47`, `2482.03`, and `2483.60 ms`. This remains
  significantly faster than the supplied llm.c baseline, confirming the
  current slowdown is specific to the pure-TK GEMM path.
- Tested pure SM120 TK codegen with `FORCE_NVCC_O=2`. It passed `test_matmul`
  (`8/8`) and `test_attention` (all three smoke shapes), but the focused
  benchmark still left pure TK behind cuBLASLt on every material GEMM row. The
  required TinyStories 3-step validation averaged `3608.73 ms` with steps
  `3604.22`, `3617.01`, and `3600.45 ms`, only a small improvement over the
  current pure-TK rebaseline and still behind the supplied llm.c baseline, so
  the build default remains `O3`.
- Promoted the pure SM120 TK huge-N forward route to a 256x128 tile by default
  (`LLMK_SM120_HUGE_N_M256=1`) while keeping the old 128x128 route available
  with `LLMK_SM120_HUGE_N_M256=0`. The A/B build passed `test_matmul` (`8/8`)
  and `test_attention` (all three smoke shapes), improved direct LM-head
  forward to `24521.92 us`, and averaged `3592.08 ms` on the required
  TinyStories 3-step validation. The promoted source-default rebuild passed the
  same smokes, kept LM-head forward at `24907.96 us` in the focused benchmark,
  and averaged `3580.59 ms` with steps `3582.89`, `3567.92`, and `3593.27 ms`.
  This is the fastest current pure-TK source default, but pure TK still trails
  the cuBLASLt-backed SM120 default and the supplied llm.c baseline.
- Changed the Makefile default so pure SM120 TK builds
  (`DEVICE_ARCH=SM120 SM120_USE_CUBLASLT_GEMM=0`) use `FORCE_NVCC_O=2`, while
  cuBLASLt-backed SM120 builds keep `O3`. On top of the accepted huge-N M256
  route, the explicit O2 build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes) and averaged `3565.63 ms` on the
  required TinyStories 3-step validation. A no-override rebuild confirmed the
  default now emits `-O2`, passed the same smokes, and averaged `3568.07 ms`
  with steps `3609.56`, `3554.65`, and `3581.49 ms`. This is faster than the
  O3/M256 source default, but pure TK still remains behind cuBLASLt and llm.c.
- Rejected a huge-N-only `LLMK_SM120_HUGE_N_SUPER_M=4` hook for the pure SM120
  TK LM-head forward path. The candidate build passed `test_matmul` (`8/8`)
  and `test_attention` (all three smoke shapes), but focused benchmarking kept
  LM-head forward at `24893.04 us` versus cuBLASLt `21835.15 us`, with dWeight
  still behind on every material row. The required TinyStories 3-step
  validation averaged `3567.69 ms` with steps `3566.36`, `3558.51`, and
  `3576.86 ms`, effectively identical to the current pure-TK default and still
  behind the supplied llm.c baseline, so the hook was removed.
- Rejected a macro-only `LLMK_SM120_HUGE_N_K_TILE=16` sweep for the pure SM120
  TK huge-N route. The first `test_matmul` pass had a transient MLP-up forward
  failure, but an immediate rerun passed `8/8` and `test_attention` passed all
  three smoke shapes. Focused benchmarking worsened LM-head forward to
  `25602.48 us` versus cuBLASLt `21762.29 us`, and the required TinyStories
  3-step validation averaged `3569.09 ms` with steps `3584.11`, `3567.63`, and
  `3570.55 ms`, so the default huge-N K tile remains `64`.
- Rejected a temporary `LLMK_SM120_DWEIGHT_REDUCE_BLOCK_SIZE=512` hook for the
  split-K BF16 partial reducer. The candidate passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark did not
  improve the dWeight rows and the required TinyStories 3-step validation
  regressed to `3605.23 ms` with steps `3612.75`, `3601.57`, and `3608.89 ms`.
  The reducer stays at its 256-thread launch.
- Rejected a temporary TN-only launch-bounds hook
  (`LLMK_SM120_TN_MIN_BLOCKS_PER_SM=2`) for the pure SM120 TK dWeight kernel.
  The build passed `test_matmul` (`8/8`) and `test_attention` (all three smoke
  shapes), but focused benchmarking worsened every dWeight row and the required
  TinyStories 3-step validation regressed to `3615.00 ms` with steps
  `3648.77`, `3618.72`, and `3611.28 ms`. The TN kernel launch bounds remain
  `__launch_bounds__(T::NUM_THREADS, 1)`.
- Rejected pure SM120 TK `FORCE_NVCC_O=1` codegen. The first `test_matmul`
  pass had a transient MLP-up forward failure, but an immediate rerun passed
  `8/8` and `test_attention` passed all three smoke shapes. The focused
  benchmark remained mixed with all material dWeight rows and LM-head forward
  still behind cuBLASLt, and the required TinyStories 3-step validation
  averaged `3581.32 ms` with steps `3616.81`, `3585.58`, and `3577.06 ms`.
  Pure SM120 TK builds therefore keep the current `FORCE_NVCC_O=2` default.
- Rejected a macro-only `LLMK_SM120_DINP_SUPER_M=2` sweep for pure SM120 TK
  dInput rows. It passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), but the focused benchmark did not improve the NN dInput rows
  and the required TinyStories 3-step validation regressed to `3585.48 ms`
  with steps `3578.11`, `3595.51`, and `3575.46 ms`. dInput keeps the global
  `LLMK_SM120_SUPER_M=8` default.
- Rejected disabling the pure SM120 TK N96 forward tile
  (`LLMK_SM120_FORWARD_N96=0`) after the O2/M256 default changes. The build
  passed `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes),
  but the focused benchmark regressed every projection forward row and the
  required TinyStories 3-step validation averaged `3644.00 ms` with steps
  `3650.67`, `3644.67`, and `3643.34 ms`. The N96 forward tile stays enabled.
- Rejected disabling pure SM120 TK fused dGELU (`LLMK_SM120_FUSE_DGELU=0`)
  after the O2/M256 default changes. The build passed `test_matmul` (`7/7`,
  with fused dGELU smoke skipped by the macro) and `test_attention` (all three
  smoke shapes), but the focused benchmark worsened the FC dInput row and the
  required TinyStories 3-step validation regressed to `3635.67 ms` with steps
  `3630.61`, `3646.37`, and `3624.97 ms`. Fused dGELU stays enabled by
  default for pure SM120 TK.
- Rejected `LLMK_SM120_BIAS_BLOCK_SIZE=1024` for the CUDA bias-gradient
  reduction under the current pure SM120 TK defaults. It passed `test_matmul`
  (`8/8`) and `test_attention` (all three smoke shapes), but the focused GEMM
  benchmark did not show useful surrounding-kernel improvement and the required
  TinyStories 3-step validation regressed to `3621.31 ms` with steps
  `3595.30`, `3623.52`, and `3619.10 ms`. The SM120 bias-gradient block size
  remains `512`.
- Rejected `LLMK_SM120_BIAS_BLOCK_SIZE=256` for the same CUDA bias-gradient
  reduction path. It passed `test_matmul` (`8/8`) and `test_attention` (all
  three smoke shapes), but the focused benchmark did not expose a useful GEMM
  side-effect and the required TinyStories 3-step validation regressed to
  `3617.33 ms` with steps `3567.63`, `3630.49`, and `3604.18 ms`. The SM120
  bias-gradient block size remains `512`.

## 2026-05-09 — Blackwell build support

- Added Makefile `DEVICE_ARCH=SM100` and `DEVICE_ARCH=SM103` targets alongside
  the existing `SM90` and `SM120` paths, using ThunderKittens 2.0's
  `KITTENS_SM100` / `KITTENS_SM103` / `KITTENS_SM120` architecture macros.
- Generalized the TK bridge guard to allow Hopper and Blackwell macros, while
  keeping BF16 locked.
- Kept Hopper on the optimized TK H100 GEMM/MHA/GQA/RoPE wrappers and added
  Blackwell CUDA correctness fallbacks for Hopper-only GEMM, GPT MHA, Llama
  GQA, and RoPE paths so GPT/Llama trainers and smoke targets compile for
  Blackwell before dedicated B200/GB200 kernels land.
- Extended the validation harness and CUDA runtime probe with a datacenter
  Blackwell target (`DEVICE_TEST_TARGET=blackwell`, `blackwell-device`) plus a
  full `blackwell-compile` phase for model and smoke binaries.

## 2026-05-09 — deep documentation refresh

- Added [`docs/cli-reference.md`](docs/cli-reference.md): consolidated flag
  tables for `train_gpt2cu`, `train_llama3cu`, the dataset prep helpers,
  `download_llama3.py`, `profile_gpt2cu.py`, the `validate_*` family, and the
  source-contract guards. Previously the only canonical CLI text was in each
  trainer's `error_usage()` block.
- Added [`docs/validation-harness.md`](docs/validation-harness.md): full phase
  catalogue, success-marker list, validate-only mode table, required
  `goal-complete` thresholds, and runnable recipes for
  [`scripts/validate_goal_h100.sh`](scripts/validate_goal_h100.sh). Includes a
  Mermaid map of `goal-core` / `goal-complete` composition. Replaces the
  previous coverage that was scattered across `build-and-run.md` and
  `testing.md`.
- Added a checkpoint/resume sequence Mermaid diagram to
  [`docs/llama3.md`](docs/llama3.md), covering `state_*_*.bin`, `model_*.bin`,
  the `DONE_*` visibility marker, `find_max_step()`, and the
  `validate_llama_checkpoint_artifacts.py` parser.
- Added a second repo-local agent skill:
  [`.claude/skills/validate-h100/SKILL.md`](.claude/skills/validate-h100/SKILL.md).
  It routes future LLMs through the harness reference, the threshold table,
  and validate-only replay paths. [`docs/agents.md`](docs/agents.md) now
  indexes both skills.
- Refreshed the top-level [`README.md`](README.md) and
  [`docs/README.md`](docs/README.md) routing tables to point at the new pages.
- Cross-linked [`docs/build-and-run.md`](docs/build-and-run.md) and
  [`docs/testing.md`](docs/testing.md) to the new harness reference instead of
  duplicating its content.
- Regenerated [`llms-full.txt`](llms-full.txt) and updated
  [`llms.txt`](llms.txt) to include the two new pages and the new skill, in a
  consistent ingestion order.

## 2026-05 — M8 profiling gate hardening

- Added an RTX 5090 generic device-test target without weakening the H100 goal
  gates. `scripts/validate_goal_h100.sh rtx5090-device` now forces
  `DEVICE_TEST_TARGET=rtx5090` and `DEVICE_ARCH=SM120`, skips NCCL/MPI by
  default, builds `cuda_runtime_check` plus the plain CUDA `test_swiglu`, and
  runs both device probes. Full TK/model-kernel runtime evidence remains
  H100-only.
- Replaced the ZeRO-3 runtime fail-fast with a compile-wired parameter-shard
  runtime path for GPT and Llama. ZeRO-3 now allocates an authoritative local
  BF16 parameter shard, initializes it from the full parameter layout, runs
  AdamW on the owned shard, and all-gathers back into the full compute layout
  used by the current forward/backward kernels. Source guards now check the
  shard-local update and all-gather contract; H100/NCCL end-to-end validation
  is still pending.
- Added [`dev/validate_zero_layout.py`](dev/validate_zero_layout.py) and wired
  it into `python-syntax` and `source-guards`. It checks host-only ZeRO local
  shard offsets for GPT-2, every built-in GPT-3 descriptor, and Llama-3 1B/8B
  across 1/2/4/8/16 processes.
- Updated the ZeRO docs/index text that still described ZeRO-3 as a runtime
  fail-fast path; current docs now describe the parameter-shard runtime path
  and the remaining H100/NCCL validation gate.
- Added the parser-supported `gpt3:c384` descriptor to the `gpt-dry` harness
  loop and coverage guard so dry-run validation matches the full built-in GPT-3
  descriptor surface.
- Extended the NCCL/ZeRO source guard to require post-update synchronization
  after ZeRO-3 parameter all-gathers in both trainers, so later full-layout
  reads cannot race the update-time all-gather.
- Strengthened the GQA/RoPE source guard so supported-shape TK backward must
  receive the RoPE tables and the wrapper must still inverse-rotate packed
  `dQ`/`dK` gradients after the TK gradient path before writing `dinp`.
- Added negative captured-evidence replay checks for `goal-complete-prereqs`.
  The host-only replay smoke now proves the completion prereq path rejects
  `ALLOW_NON_H100`, missing explicit thresholds, missing GQA runtime markers,
  missing ZeRO-3 smoke stage evidence, missing GPT-2 full-run launch evidence,
  and missing profile CSV evidence.
- Hardened full-run evidence: GPT-2 124M, Llama-3 1B, and Llama-3 8B launch
  scripts now write `run.log` metadata, and the harness requires that metadata
  plus final checkpoint markers/artifacts for validate-only completion checks.
- Hardened profile completion evidence: `goal-complete` now forces/requires
  both `profile_ge0` and `profile_ge1` artifacts, while standalone `profile`
  runs may still narrow `PROFILE_GELU_FUSIONS` for debugging.
- Hardened Llama-3 1B stability completion evidence: `goal-complete` now forces
  HellaSwag on for the stability phase, and any
  `LLAMA1B_STABILITY_MIN_HELLASWAG` threshold requires final-step eval evidence.
- Added `ZERO3_SMOKE_MAX_VAL_LOSS` to the ZeRO-3 GPT-2 runtime smoke verifier
  and the required `goal-complete` threshold set, so ZeRO-3 completion evidence
  includes an explicit final validation-loss ceiling.
- Hardened validate-only ZeRO-3 smoke evidence: live `zero3-smoke` now writes a
  run log, validate-only mode requires `ZERO3_SMOKE_RUN_LOG`, and prereq replay
  rejects logs that do not contain the ZeRO-3 stage banner.
- Added fail-fast validation for `goal-complete` metric thresholds. Loss and
  tolerance thresholds must be positive finite numbers, HellaSwag thresholds
  must be in `[0,1]`, and the replay smoke covers malformed threshold failures.
- Cleaned up H100 compile-log noise in the local TK wrappers: MHA/GQA attention
  and RoPE now cast CUDA grid indices before constructing ThunderKittens
  coordinates, and the MFU helper uses non-deprecated NVML temperature/clock
  event APIs. The full compile harness now rebuilds the local CUDA targets
  without the previous narrowing/deprecation warning flood, and
  `dev/validate_build_contracts.py` guards those warning-clean source
  contracts.
- Extended [`profile_gpt2cu.py`](profile_gpt2cu.py) with explicit CLI controls
  for the profiling binary, output report, build/run skipping, and the minimum
  averaged tensor-core utilization threshold. The default threshold is 70%, so
  the helper now fails the M8 gate instead of only printing the metric.
- Wired `PROFILE_MIN_TENSOR_UTIL` through
  [`scripts/validate_goal_h100.sh`](scripts/validate_goal_h100.sh) for the
  `profile` phase. The real H100 `ncu` run is still pending in this workspace
  because CUDA runtime access fails before model code.
- Added explicit `--gelu-fusion 0|1` profiling support to
  [`profile_gpt2.cu`](profile_gpt2.cu) and
  [`profile_gpt2cu.py`](profile_gpt2cu.py), and made the `profile` harness run
  `PROFILE_GELU_FUSIONS="0 1"` by default so the eventual H100 profile gate
  covers both the default GPT-2 MLP path and the opt-in TK bias+GELU epilogue.
- Added `LLAMA_DRY_CHECKPOINT` / `LLAMA_DRY_ZERO_STAGE` to the `llama-dry`
  validation phase so a converted gated Llama checkpoint can be checked by the
  host-only C++ parser and ZeRO layout validator before CUDA/NCCL startup.
- Added [`dev/validate_gpt2_starter_pack.py`](dev/validate_gpt2_starter_pack.py)
  and wired it into the `starter-pack` phase. It validates GPT-2 fp32/BF16
  checkpoint headers and sizes, tokenizer header/token payload, and
  `gpt2_124M_debug_state.bin` shape, token range, expected loss, sampled
  logits/gradients, and exact byte count without initializing CUDA. The phase
  now runs `--self-test` first with tiny synthetic starter-pack artifacts and
  expected parser failures.
- Added a `script-syntax` harness phase and included it in `goal-core`, covering
  the launch, multi-node, data-download, and starter-pack shell scripts with
  `bash -n`.
- Added a `python-syntax` harness phase and included it in `goal-core`, covering
  the dataset, converter, profiling, and starter-pack Python helpers with
  `python3 -m py_compile`.
- Added a `host-core` harness aggregate for local machines without a usable CUDA
  runtime. It runs the non-CUDA-runtime host-side gates against existing built
  binaries and artifacts; `all-local` now aliases this phase.
- Tightened `scripts/validate_goal_h100.sh preflight` so H100 runtime gates
  require H100/sm90-class GPUs. Unsupported devices fail before NCCL/MPI checks
  unless `ALLOW_NON_H100=1` is set for dry compile/debug runs. RTX 5090 now has
  a separate generic device-test path, but it is not H100 runtime evidence.
- Matched the standalone `cuda-runtime` probe to the same target contract, so
  running that phase directly cannot accept the wrong GPU class as runtime
  evidence.
- Made `goal-complete` reject `ALLOW_NON_H100=1`, keeping the dry-debug escape
  hatch out of the one-shot completion gate.
- Added [`dev/validate_data_artifacts.py`](dev/validate_data_artifacts.py) and a
  `data-artifacts` harness phase. It validates prepared GPT-2/Llama training
  and HellaSwag-style eval `.bin` headers, exact file sizes, token widths,
  sampled train-token ranges, and eval-example streams without CUDA. The phase
  now runs `--self-test` first, covering synthetic GPT/Llama train/eval artifacts
  and expected parser failures before checking real prepared data.
- Added [`dev/test_dataloader.cpp`](dev/test_dataloader.cpp), `make
  test_dataloader`, and a `dataloader-smoke` harness phase. The smoke writes
  synthetic GPT-2 uint16 and Llama-3 uint32 train/eval files under `/tmp` and
  checks the host-side `DataLoader`/`EvalLoader` dispatch, rank offsets, shifted
  targets, labels, and masks.
- Added `dev/download_llama3.py --write-synthetic-checkpoint` and a
  `llama-checkpoint-smoke` harness phase. The phase writes a tiny deterministic
  BF16 Llama checkpoint, validates it with the Python and host-only C++ parsers,
  then checks the 8-process ZeRO-2 layout without gated HF weights or CUDA
  initialization.
- Tightened the Llama host-only dry-run path to use the shared
  `set_zero_configs` helper instead of hand-filling ZeRO fields, so `llama-dry`
  now validates the same local shard parameter count used by runtime.
- Added [`dev/validate_attention_gqa_reference.py`](dev/validate_attention_gqa_reference.py)
  and a `gqa-reference` harness phase. It checks the `B=1 T=128` and `B=1
  T=256` Llama GQA/RoPE smoke shapes on CPU by comparing materialized-RoPE
  repeated-KV attention with grouped/tile-load-style RoPE, including backward
  gradients into packed Q/K/V.
- Added `profile_gpt2cu.py --csv-input`,
  [`dev/validate_profile_parser.py`](dev/validate_profile_parser.py), and a
  `profile-parser` harness phase. The validator feeds synthetic Nsight Compute
  raw CSV into the parser and checks both passing and failing tensor-utilization
  thresholds without requiring `ncu`.
- Added [`dev/validate_log_tools.py`](dev/validate_log_tools.py) and a
  `log-tools` harness phase. It feeds synthetic rank-0 logs through
  `dev/validate_training_log.py` and `dev/compare_training_logs.py`, checking
  both passing cases and expected threshold, expected-metric tolerance,
  final-step, and loss-curve failures without launching training.
- Added `dev/validate_llama_checkpoint_artifacts.py --self-test` and wired it
  into `llama-checkpoint-smoke`, so model/state artifact header validation has
  a synthetic pass/fail check before real resume outputs exist.
- Added [`dev/validate_llama3_converter.py`](dev/validate_llama3_converter.py)
  and a `llama-converter-smoke` harness phase. It fills a tiny Llama model with
  deterministic BF16 values, runs `train_llama3.py::write_model`, verifies the
  header and payload tensor order, and dry-parses the result with
  `train_llama3cu -x 0 -z 2` without gated HF weights.
- Added a `gqa-runtime` harness phase that runs the CPU-only
  `dev/validate_attention_gqa_reference.py` check and then executes
  `test_attention_gqa` as the dedicated H100 CUDA/TK comparison for the
  `B=1 T=128` and `B=1 T=256` Llama GQA/RoPE smoke shapes.
- Strengthened `gqa-runtime` so it asserts explicit per-shape markers for the
  `T=128` fallback-backward case and the `T=256` TK-backward/tile-RoPE case,
  not just the final smoke marker.
- Switched `scripts/run_llama3_1B.sh` defaults to Llama-tokenized
  FineWeb-edu 100B paths and added a bounded `llama1b-stability` harness phase
  for the 1000-step M6 stability gate, with HellaSwag eval required by default.
- Fixed scalar NCCL all-reduce call sites in `llmc/zero.cuh`,
  `train_gpt2.cu`, and `train_llama3.cu` to pass an element count of `1`
  instead of `sizeof(float)`, which would otherwise overrun single-float device
  buffers. Added `dev/validate_nccl_source.py` and the `source-guards` harness
  phase to keep that contract checked without launching NCCL.
- Added `multi_gpu_sync_nccl_stream_from_compute()` and used it before ZeRO
  optimizer shard `ncclAllGather` calls in the GPT and Llama update paths, so
  NCCL waits for AdamW kernels on the compute stream before reading updated
  parameter shards. The source guard now checks this ordering contract too.
- Hardened NCCL build discovery in the Makefile. Multi-GPU builds now detect
  standard NCCL installs through `ldconfig` plus `nccl.h`, and cluster/module
  installs can be selected with `NCCL_DIR`, `NCCL_INCLUDE_PATH`, and
  `NCCL_LIB_PATH` instead of depending only on `dpkg` package metadata. An
  explicit include/library path pair is enough; `NCCL_DIR` is not required.
- Aligned `scripts/validate_goal_h100.sh preflight` with that NCCL discovery
  path so the H100 gate validates the same system or custom NCCL install that
  the Makefile will compile against.
- Added host-only ZeRO-3 layout validation for GPT and Llama dry-runs. `-x 0
  -z 3` now checks tensor divisibility and local shard counts, while `-x >0
  -z 3` still fails before CUDA/NCCL startup because runtime parameter
  all-gather/scatter is not implemented.
- Extended `dev/validate_nccl_source.py` so `source-guards` also checks the
  explicit ZeRO-3 runtime diagnostic, the current full parameter/gradient
  trainer residency, and that the ZeRO-3 runtime rejection remains after
  host-only dry-runs but before `multi_gpu_config_init`. This prevents
  `-x >0 -z 3` from reaching CUDA/NCCL startup until parameter
  all-gather/scatter is implemented.
- Strengthened `scripts/validate_goal_h100.sh zero-guards` so negative ZeRO
  cases must fail with the intended diagnostic text, and added GPT/Llama
  `-z 4` checks for unsupported-stage rejection.
- Strengthened the `gpt-dry` and `llama-dry` harness phases with positive
  output assertions for descriptor/layout evidence, including GPT-2 ZeRO-1/3,
  every built-in GPT-3 descriptor's source/channel/ZeRO-2 markers, and Llama-3
  1B/8B/3.1 8B source plus ZeRO layout markers.
- Added `--cpp-zero-stage` and `--cpp-processes` to
  `dev/download_llama3.py --cpp-validate`, and routed the synthetic Llama
  checkpoint smoke through those options so converter-backed C++ dry-runs can
  validate ZeRO layout directly.
- Converted the GPT-2 starter-pack and Llama converter smoke phases to assert
  stable success markers instead of relying only on command exit status.
- Converted the data artifact, dataloader smoke, GQA reference, and profile
  parser host-only phases to assert their final success markers.
- Added final success markers to the CUDA smoke/parity binaries and made the
  H100 harness assert them: `<binary> smoke OK` for the kernel smokes,
  `test_attention_gqa smoke OK` for `gqa-runtime`, `CUDA runtime check passed.`
  for `cuda-runtime`, and `gpt2_validate OK` / `test_gpt2cu OK` for the GPT-2
  gates.
- Added `dev/validate_runtime_markers.py` to `source-guards` so the CUDA
  runtime, kernel-smoke, GPT-2 validation, and GQA runtime success-marker
  contracts are checked without launching CUDA.
- Added `dev/validate_goal_harness_coverage.py` to `source-guards` so compile
  target coverage, `goal-complete` phase coverage, and required explicit metric
  thresholds are checked against `goal.md` without launching long jobs.
- Extended `dev/validate_goal_harness_coverage.py` with a runtime-evidence map
  for the remaining unchecked `goal.md` gates, tying each one to the concrete
  harness phase, success marker, log verifier, profile mode, or conversion
  validator that must pass before the goal can be claimed complete.
- Expanded that runtime-evidence map to explicitly cover ZeRO-2 GPT/Llama
  dry-run layouts, ZeRO-3 fail-fast diagnostics, and the Llama-3 8B multi-node
  full-run artifact/log checks.
- Added a guard that fails when a new unchecked `goal.md` `- [ ]` item appears
  without a matching runtime-evidence mapping.
- Added `dev/validate_build_contracts.py` to `source-guards` so the BF16-only,
  H100 `sm_90a`, ThunderKittens include/define, dynamic shared-memory, and
  empty cuBLAS-shim contracts are source-checked before runtime gates.
- Added `dev/validate_epilogue_source.py` to `source-guards` so the optional
  GPT-2 MLP bias+GELU epilogue remains aligned across the TK GEMM template,
  matmul wrapper, `-ge` switch/fallback, profile switch, larger launch scripts, and
  `test_matmul` smoke coverage.
- Added `dev/validate_gqa_source.py` to `source-guards` so the custom
  GQA/RoPE tile-load routing, query-to-KV head mapping, supported-shape gates,
  and T=128/T=256 smoke/reference coverage are source-checked.
- Added `dev/validate_training_source.py` to `source-guards` so the rank-0
  `main.log` format, trainer logger initialization, and harness log-validation
  arguments are source-checked against `dev/validate_training_log.py`.
- Added `dev/validate_profile_source.py` to `source-guards` so the
  `profile_gpt2cu.py` ncu command, raw metrics, tensor-core utilization gate,
  profiling binary, parser smoke, and harness profile phase stay aligned.
- Added `dev/validate_llama_conversion_source.py` to `source-guards` so the
  gated Llama-3.1 8B HF alias, BF16 checkpoint validation, synthetic checkpoint
  path, C++ dry-parse options, and `llama8b-convert` phase stay aligned.
- Added ZeRO-2 impossible-process-count checks to `zero-guards` for GPT and
  Llama dry-runs, asserting the partitioning diagnostic before CUDA/NCCL init.
- Converted `source-guards` to assert the `NCCL/ZeRO source guards OK` success
  marker.
- Strengthened optional `GPT_DRY_CHECKPOINT` and `LLAMA_DRY_CHECKPOINT`
  branches so checkpoint dry-runs assert the expected parser/layout output.
- Added a guarded `goal-complete` harness phase. With
  `ALLOW_FULL_GOAL_RUN=1`, it runs `goal-core` plus the long
  H100/NCCL/profile/conversion and full-run gates in one explicit completion
  pass.
- Made `goal-complete` fail fast on required completion tooling and artifacts:
  `ncu`, `gpt2_124M_bf16.bin`, and `sbatch` when the two-node/full 8B phases
  are not in validate-only mode.
- Added validate-only evidence preflight to `goal-complete`, so existing-log
  two-node reference/candidate checks and existing-artifact 8B full checks
  prove their required files before `goal-core` starts.
- Exposed those completion prerequisite checks as
  `scripts/validate_goal_h100.sh goal-complete-prereqs` for a no-launch
  operator preflight.
- Added `GPT2_FULL_VALIDATE_ONLY=1` and `LLAMA1B_FULL_VALIDATE_ONLY=1` so
  completed single-node full-run evidence can be validated without relaunching.
- Added `PROFILE_VALIDATE_ONLY=1` profile replay. `PROFILE_CSV_DIR` validates
  existing raw `profile_ge*.csv` exports without Nsight Compute on the
  validation host; `PROFILE_REPORT_DIR` validates existing `profile_ge*.ncu-rep`
  reports when local `ncu` is available to export the raw CSV.
- Added captured-log replay for short runtime gates:
  `PREFLIGHT_VALIDATE_ONLY`, `CUDA_RUNTIME_VALIDATE_ONLY`, `SMOKE_VALIDATE_ONLY`,
  `GPT2_RUNTIME_VALIDATE_ONLY`, `GQA_RUNTIME_VALIDATE_ONLY`,
  `GPT2_SMOKE_VALIDATE_ONLY`, `LLAMA_RESUME_VALIDATE_ONLY`, and
  `LLAMA1B_STABILITY_VALIDATE_ONLY`.
- Added `dev/validate_goal_replay.py` and the `goal-replay-smoke` harness phase
  to exercise captured-evidence replay with synthetic logs/artifacts.
- Added `LLAMA8B_CONVERT_VALIDATE_ONLY=1` so evidence-only completion checks
  require an existing 8B checkpoint instead of attempting a gated HF conversion.
- Added a `llama8b-convert` harness phase for the real gated HF Llama-3.1 8B
  converter gate. It validates an existing `LLAMA8B_CHECKPOINT` or converts
  `${LLAMA8B_MODEL:-llama3.1:8B}`, then dry-parses the checkpoint through
  ZeRO-2/16-process C++ layout validation by default.
- Added [`dev/validate_training_log.py`](dev/validate_training_log.py), a
  host-only rank-0 `main.log` verifier for long training gates. It parses
  validation loss, HellaSwag/eval accuracy, and train loss/LR/grad-norm lines;
  checks final steps, finite metrics, optional published/threshold values, and
  train-loss decrease where required.
- Wired GPT-2-style `main.log` logging into [`train_llama3.cu`](train_llama3.cu)
  for validation, eval, and train metrics, matching the existing GPT-2 logger
  format.
- Hardened `llama1b-stability`, `gpt2-full`, and `llama1b-full` so they run
  `dev/validate_training_log.py` after launch instead of treating process exit
  as sufficient evidence. Llama phases require train-loss decrease; GPT-2 full
  can compare against `GPT2_FULL_EXPECTED_VAL_LOSS` and
  `GPT2_FULL_EXPECTED_HELLASWAG`.
- Hardened `gpt2-smoke` so the tiny-shakespeare smoke run also validates
  `main.log` after launch, requiring final validation/train metrics and
  train-loss decrease. `GPT2_SMOKE_MAX_VAL_LOSS` can add a target-host
  validation-loss ceiling.
- Hardened `llama-resume` so the checkpoint/restart smoke validates the initial
  and final `DONE_*`, model, and rank-0 state files, then validates `main.log`
  after the resumed run. Added
  [`dev/validate_llama_checkpoint_artifacts.py`](dev/validate_llama_checkpoint_artifacts.py)
  to parse model/state headers and check magic, version, step, rank, and
  process count without CUDA. `LLAMA_RESUME_MAX_VAL_LOSS` can add a target-host
  validation-loss ceiling.
- Strengthened `goal-complete` so it fails fast unless
  `GPT2_FULL_EXPECTED_VAL_LOSS`, `GPT2_FULL_EXPECTED_HELLASWAG`, and the
  smoke/Llama max-loss/min-HellaSwag thresholds are set, forcing completion
  runs to compare the long-run evidence against explicit target metrics.
- Hardened `llama8b-full` so the M7 Slurm gate uses `sbatch --wait` and then
  validates the final checkpoint headers plus rank-0 `main.log` metrics instead
  of accepting job submission as sufficient evidence. `LLAMA8B_FULL_VALIDATE_ONLY=1`
  checks an already completed output directory.
- Changed GPT-2 and Llama training log initialization to append only when a
  completed checkpoint is actually found, not merely when `-y 1` was requested.
  A fresh run in a stale output directory now clears `main.log` before writing
  new validation evidence.
- Added [`dev/compare_training_logs.py`](dev/compare_training_logs.py) and a
  `gpt2-two-node` harness phase for the M5 two-node sanity gate. It compares
  the first 100 paired train-loss steps from single-node and two-node rank-0
  logs using an explicit tolerance and now requires both compared train-loss
  curves to decrease over the selected window.
- Added `MAX_STEPS` overrides to the GPT-2 multi-node MPI/FS/TCP scripts so
  the two-node sanity gate can run a bounded 100-step job instead of requiring
  a full reproduction.
- Routed `MAX_STEPS` through the GPT-2 124M/350M/774M/1558M, GPT-3 125M,
  PyTorch GPT-2 124M reference, and Llama-3 1B full-run scripts and their
  harness phases. The CUDA scripts' `DONE_*` guard now derives from the same
  step count passed to `-x`, avoiding mismatches between loop completion and log
  validation.
- Added [`dev/validate_launch_scripts.py`](dev/validate_launch_scripts.py) to
  `source-guards` so the `MAX_STEPS` / `-x` / `DONE_*` launch-script contract
  is checked without submitting jobs.

## 2026-05 — M5 GPT dry-run metadata validation

- Split GPT model metadata loading from CUDA allocation in
  [`train_gpt2.cu`](train_gpt2.cu), so `train_gpt2cu -x 0` can parse GPT-2/GPT-3
  descriptors or checkpoint headers, calculate payload sizes, and validate
  ZeRO tensor shardability before CUDA/NCCL init.
- Fixed the GPT-3 13B descriptor shape to canonical `gpt3:c5120`; the inherited
  `c5140` value could not divide by the 128-wide attention head size.
- Added the `gpt-dry` phase to
  [`scripts/validate_goal_h100.sh`](scripts/validate_goal_h100.sh), covering
  GPT-2 ZeRO-1, optional GPT checkpoint header/payload validation through
  `GPT_DRY_CHECKPOINT`, and GPT-3 `c768` through `c12288` ZeRO-2 8-process
  host-only layout validation.
- Added a `starter-pack` phase that checks the GPT-2 starter-pack files are
  present and validates the BF16 checkpoint header/payload through the host-only
  GPT dry-run path.
- Added [`dev/cuda/cuda_runtime_check.cu`](dev/cuda/cuda_runtime_check.cu) and
  a `cuda-runtime` harness phase, so driver/runtime/device-allocation failures
  are reported before the heavier GPT-2 model gates.
- Verification: `make -B train_gpt2cu FORCE_NVCC_O=0 NO_MULTI_GPU=1
  NO_USE_MPI=1`, `scripts/validate_goal_h100.sh starter-pack`,
  `scripts/validate_goal_h100.sh gpt-dry`, and a real starter-pack
  `GPT_DRY_CHECKPOINT=gpt2_124M_bf16.bin` dry-run pass locally.
  `make -B cuda_runtime_check FORCE_NVCC_O=0 NO_MULTI_GPU=1 NO_USE_MPI=1`
  passes; `scripts/validate_goal_h100.sh cuda-runtime` now fails locally with
  the expected CUDA driver/runtime mismatch.

## 2026-05 — M7 Llama checkpoint validation hooks

- Wired `-z 2` through the sharded optimizer/reduce-scatter path in
  [`llmc/zero.cuh`](llmc/zero.cuh), `train_gpt2.cu`, and `train_llama3.cu`.
  `-z 3` still fails fast instead of silently falling back to ZeRO-0, and the
  real H100/NCCL ZeRO-2 run remains pending.
- Fixed [`train_llama3.py`](train_llama3.py) so `write_model()` emits the same
  hidden-dim header field as the C++ Llama checkpoint writer.
- Extended [`dev/download_llama3.py`](dev/download_llama3.py) with post-write
  validation for Llama checkpoint magic/version, expected bf16 payload size,
  and hidden-dim metadata, plus `--validate-only` for existing files.
- Added optional `--cpp-validate`, which runs `train_llama3cu -e CHECKPOINT -x 0`
  to exercise the C++ checkpoint parser and payload-size validator without
  initializing CUDA.
- Added GPT-2-style checkpoint state to [`train_llama3.cu`](train_llama3.cu):
  rank 0 writes the model, each rank writes AdamW/RNG/dataloader state, and
  `-y 1` resumes from the newest completed `DONE_*` checkpoint.
- Added [`scripts/validate_goal_h100.sh`](scripts/validate_goal_h100.sh), an
  executable target-host checklist for the remaining `goal.md` runtime gates:
  H100/CUDA/NCCL/MPI preflight, compile, kernel smoke tests, GPT-2
  validation/parity, GPT/Llama dry-run/resume smoke, host-only ZeRO layout
  dry-run checks, ZeRO runtime fail-fast guards, profiling, and full-run phases
  with final artifact/log validation.
- Verification: `python3 -m py_compile dev/download_llama3.py train_llama3.py`
  passes. A synthetic Llama checkpoint passes both
  `python3 dev/download_llama3.py --validate-only ... --cpp-validate` and the
  C++ dry-run. The real gated HF 8B conversion/load remains pending.

## 2026-05 — M3 GPT-2 parity tolerance table

- Replaced the anonymous gradient-threshold array in [`test_gpt2.cu`](test_gpt2.cu)
  with an explicit `kGradientTolerances` table that records tensor names,
  inherited llm.c BF16 thresholds, current TK thresholds, and notes for tensors
  likely to move after the first H100 TK MHA-bwd parity run.
- Fixed the inherited `attrpojw` label typo to `attprojw` so parity output maps
  cleanly to `ParameterTensors`.
- Verification: `make test_gpt2cu NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0`
  compiles. Runtime parity remains blocked locally by the CUDA driver/runtime
  mismatch.

## 2026-05 — M8 GEMM bias+GELU epilogue compile path

- Extended [`llmc/tk/gemm_h100.cuh`](llmc/tk/gemm_h100.cuh) with opt-in
  finish-path bias+GELU aliases for the `A*B^T` forward path, including a
  pre-GELU auxiliary TMA store for backward compatibility with llm.c's fused
  GELU path.
- Added [`llmc/matmul.cuh`](llmc/matmul.cuh)::`matmul_forward_gelu` and wired
  GPT-2's MLP up-projection to use it behind `train_gpt2cu -ge 1`. The trainer
  default remains `-ge 0` until H100 numerical validation passes.
- Extended [`dev/cuda/test_matmul.cu`](dev/cuda/test_matmul.cu) with a
  CPU-reference smoke case for the fused pre-GELU and GELU outputs.
- Verification: `make test_matmul train_gpt2cu test_gpt2cu gpt2_validate
  profile_gpt2cu NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0` compiles.
  `./test_matmul` and `./train_gpt2cu -x 0 -ge 1` are still blocked locally by
  the CUDA driver/runtime mismatch. H100 numerical validation and `ncu`
  profiling remain M8 gates.

## 2026-05 — M6 GQA tile-load RoPE compile path

- Added an optional tile-load RoPE path to
  [`llmc/tk/attention_gqa_h100.cuh`](llmc/tk/attention_gqa_h100.cuh): for
  shapes where TK forward and TK backward are both available, Q/K are saved
  unrotated and rotated inside the TK shared tiles before WGMMA.
- Extended the shared backward launcher in
  [`llmc/tk/attention_h100.cuh`](llmc/tk/attention_h100.cuh) so GQA backward can
  rotate Q/K tiles without changing the GPT MHA call sites.
- Fallback GQA shapes still use the fused Q/K materialization and packed-gradient
  unpermute path, preserving the existing `T=128` coverage while `T=256` now
  compile-wires the tile-load RoPE path.
- Verification: `make test_attention_gqa train_llama3cu NO_MULTI_GPU=1
  NO_USE_MPI=1 FORCE_NVCC_O=0` and `make test_attention train_gpt2cu
  test_gpt2cu NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0` compile. `./train_llama3cu
  -x 0` passes. `./test_attention_gqa` is still blocked locally by the CUDA
  driver/runtime mismatch.

## 2026-05 — M2 forward-only GPT-2 validation target

- Added [`dev/cuda/gpt2_validate.cu`](dev/cuda/gpt2_validate.cu), a focused
  forward-only gate that loads `gpt2_124M_debug_state.bin`, calls
  `gpt2_validate()`, compares the mean loss against the saved PyTorch reference
  loss, and exits before backward/AdamW.
- Added `make gpt2_validate` to the top-level [`Makefile`](Makefile) and
  documented it in the build/testing docs.
- Verification: `make gpt2_validate NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0`
  compiles. Runtime execution is still blocked locally by the CUDA
  driver/runtime mismatch.

## 2026-05 — M6 GQA RoPE materialization fusion

- Added fused Q/K materialization kernels in [`llmc/attention_gqa.cuh`](llmc/attention_gqa.cuh)
  so forward RoPE is applied while unpacking packed Llama Q/K/V. This removes
  the standalone forward `rope_forward` launches before GQA attention while
  preserving the rotated `qkvr` layout that backward expects.
- Added a RoPE-aware packed-gradient unpermute kernel so inverse RoPE is
  applied while writing Q/K gradients back to packed input-gradient layout,
  removing the standalone backward `rope_backward` launches from GQA attention.
- Updated the M6 docs to distinguish this landed materialization fusion from
  the still-pending final RoPE fusion inside the TK tile-load path.
- Verification: `make test_attention_gqa NO_MULTI_GPU=1 NO_USE_MPI=1
  FORCE_NVCC_O=0` compiles. Runtime execution is still blocked locally by the
  CUDA driver/runtime mismatch.

## 2026-05 — M2/M3 LayerNorm smoke harness

- Added [`dev/cuda/test_layernorm.cu`](dev/cuda/test_layernorm.cu), a GPT-style
  LayerNorm smoke harness with independent CPU references for forward, fused
  residual+LayerNorm forward, saved `mean`/`rstd`, and backward `+=`
  accumulation into `dinp`, `dweight`, and `dbias`.
- Added `make test_layernorm` to the top-level [`Makefile`](Makefile) and
  documented it across the testing/build/kernel-reference docs.
- Verification: `make test_layernorm NO_MULTI_GPU=1 NO_USE_MPI=1
  FORCE_NVCC_O=0` compiles. Runtime execution is still blocked locally by the
  CUDA driver/runtime mismatch.

## 2026-05 — M2/M3 GPT MHA smoke harness

- Added [`dev/cuda/test_attention.cu`](dev/cuda/test_attention.cu), a GPT-style
  MHA smoke harness with an independent CPU reference for packed Q/K/V causal
  forward and packed Q/K/V input gradients.
- The harness covers direct TK forward plus CUDA fallback backward at `T=192`,
  and padded TK forward plus supported-shape TK backward at `T=256`.
- Added `make test_attention` to the top-level [`Makefile`](Makefile) and
  documented it across the testing/build/kernel-reference docs.
- Verification: `make test_attention NO_MULTI_GPU=1 NO_USE_MPI=1
  FORCE_NVCC_O=0` compiles. Runtime execution is still blocked locally by the
  CUDA driver/runtime mismatch.

## 2026-05 — M4/M5 launch-script ports

- Added GPT-2/GPT-3 launch scripts under [`scripts/`](scripts/):
  `run_gpt2_124M.sh`, `run_gpt2_350M.sh`, `run_gpt2_774M.sh`,
  `run_gpt2_1558M.sh`, `run_gpt3_125M.sh`, and `pyrun_gpt2_124M.sh`.
- Added multi-node GPT-2 124M launch scripts under [`scripts/multi_node/`](scripts/multi_node/):
  MPI, filesystem rendezvous, and TCP rendezvous variants.
- Added the upstream [`train_gpt2.py`](train_gpt2.py) PyTorch reference helper
  for the PyTorch run script and reference `.bin` generation path.
- Each distributed script documents H100 NCCL defaults inline:
  `NCCL_NVLS_ENABLE=1`, `NCCL_IB_HCA=mlx5`, `NCCL_NET_GDR_LEVEL=2`, and
  `NCCL_IB_DISABLE=0`. Scripts syntax-check locally; H100/NCCL runtime parity
  is still pending.

## 2026-05 — M8 profiling binary

- Added [`profile_gpt2.cu`](profile_gpt2.cu), adapted from llm.c's profiling
  helper. It includes `train_gpt2.cu` under `TESTING`, runs one GPT-2
  forward/backward/update step, and uses a single-process filesystem NCCL init
  path when compiled with multi-GPU support.
- Adapted [`profile_gpt2cu.py`](profile_gpt2cu.py) to build the profiling
  binary with the repo's TK-only `NO_MULTI_GPU=1 NO_USE_MPI=1` path instead of
  stale llm.c cuDNN flags, and to tolerate hosts where `modprobe -c nvidia`
  cannot be inspected before trying `ncu`.
- `make profile_gpt2cu NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0` compile-checks
  successfully. The actual `ncu` run and utilization threshold remain pending
  until H100 runtime access is available.

## 2026-05 — M8 tutorial archive

- Added [`doc/`](doc/) as the narrative "how this kernel was ported" archive,
  separate from the operational [`docs/`](docs/) tree.
- Added tutorial pages for GEMM, attention, normalization, and Llama-3:
  [`doc/gemm/gemm.md`](doc/gemm/gemm.md),
  [`doc/attention/attention.md`](doc/attention/attention.md),
  [`doc/norms/norms.md`](doc/norms/norms.md), and
  [`doc/llama3/llama3.md`](doc/llama3/llama3.md).
- M8 remains partial: the tutorial archive and profiling binary compile path
  exist, and the optional TK GEMM epilogue is now compile-wired behind `-ge 1`,
  but the real H100 `ncu` run and epilogue numerical validation are still
  pending.

## 2026-05 — M2/M3 GEMM layout correction

- Extended [`llmc/tk/gemm_h100.cuh`](llmc/tk/gemm_h100.cuh) with `A*B^T`
  specializations so `matmul_forward` consumes llm.c checkpoint weights in
  their real `(OC, C)` layout instead of the synthetic `(C, OC)` smoke-test
  layout.
- Updated [`dev/cuda/test_matmul.cu`](dev/cuda/test_matmul.cu) so the reference
  path now stores weights as `(N, K)` and checks `A * W^T`, matching model use.
- Corrected matmul backward baseline indexing for `(OC, C)` weights, moved
  `dinp` backward onto the existing TK `A*B` GEMM path, and wired `dbias`
  through the verbatim llm.c reduction kernels when the auxiliary buffer is
  available. At this point, `dweight` still remained the matmul M3 TK task;
  the follow-up entry below adds the non-accumulating TK path.
- Verification: `make all NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0` compiles
  `test_matmul`, `train_gpt2cu`, and `test_gpt2cu`. Runtime H100 validation is
  still pending.

## 2026-05 — M3 partial TK dWeight path

- Extended [`llmc/tk/gemm_h100.cuh`](llmc/tk/gemm_h100.cuh) with
  `A_TRANSPOSED` specializations and `mma_AtB` dispatch for `A^T*B`.
- Updated [`llmc/matmul.cuh`](llmc/matmul.cuh) so dWeight backward uses TK
  `A^T*B` when the destination gradient buffer is known to be zero, and uses
  a caller-provided scratch buffer plus a small add kernel for accumulated
  `dWeight += ...` microsteps. The slow CUDA `+=` kernel remains only as a
  fallback for unsupported shapes or missing scratch.
- Updated [`train_gpt2.cu`](train_gpt2.cu) with a dedicated aligned
  `matmul_scratch` activation buffer and 128-byte activation-tensor alignment
  in the shared activation allocator.
- Extended [`dev/cuda/test_matmul.cu`](dev/cuda/test_matmul.cu) with direct
  dWeight `A^T*B` smoke cases for both overwrite and accumulated `+=` paths
  against naive references.
- Verification: `make all NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0` and
  `make profile_gpt2cu NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0` compile;
  `make test_matmul NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0` compiles after
  the smoke-test extension.

## 2026-05 — M3 LayerNorm backward TK reductions

- Updated [`llmc/layernorm.cuh`](llmc/layernorm.cuh) so
  `layernorm_backward_kernel10` keeps the llm.c cross-block atomic-counter
  accumulator pattern while replacing the row-wise warp reductions with a TK
  `kittens::warp::sum` shared-vector helper.
- Increased the backward kernel dynamic shared-memory opt-in to account for
  the per-warp TK reduction scratch.
- Verification: `make train_gpt2cu NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0`
  and `make test_gpt2cu NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0` compile.
  Runtime H100 parity is still pending.

## 2026-05 — M3 TK MHA backward path

- Ported the TK H100 MHA backward prep and main kernels into
  [`llmc/tk/attention_h100.cuh`](llmc/tk/attention_h100.cuh):
  `bwd_attend_prep_ker` and `bwd_attend_ker`.
- Updated [`llmc/attention.cuh`](llmc/attention.cuh) so `attention_backward`
  dispatches to TK for GPT-style `head_dim ∈ {64, 128}` with `T % 256 == 0`,
  using the forward-saved LSE in `att` and the saved forward output as TK's
  `o` input. Unsupported backward shapes still use the slow CUDA recompute
  fallback.
- Verification: `make train_gpt2cu`, `make test_gpt2cu`, `make profile_gpt2cu`,
  and `make test_matmul` compile with `NO_MULTI_GPU=1 NO_USE_MPI=1
  FORCE_NVCC_O=0`. Runtime H100 parity is still pending.

## 2026-05 — M6 Llama Python converter path

- Added [`train_llama3.py`](train_llama3.py), copied from llm.c's Llama-3
  PyTorch reference/converter helper. It includes the HF loader and `.bin`
  writer used by the future C++ Llama trainer.
- Added [`dev/download_llama3.py`](dev/download_llama3.py), a small wrapper for
  `python dev/download_llama3.py llama3.1:8B` that writes
  `llama3.1_8B_bf16.bin` from `meta-llama/Meta-Llama-3.1-8B`.
- The files syntax-check locally. Real conversion still requires HF gated-model
  access and enough GPU memory to load the 8B model.

## 2026-05 — M6 SwiGLU primitive

- Added [`llmc/swiglu.cuh`](llmc/swiglu.cuh), a plain CUDA forward/backward
  implementation for Llama-3's `out = silu(gate) * up` activation.
- Compile-checked the header with `nvcc -x cu -c llmc/swiglu.cuh`; integration
  into `train_llama3.cu` remains pending with the rest of M6.

## 2026-05 — M6 Llama training dataloader dispatch

- Extended [`llmc/dataloader.h`](llmc/dataloader.h) to detect training shard
  format from the header: GPT-2 remains magic `20240520` v1 with uint16 tokens,
  and Llama-3 uses magic `20240801` v7 with uint32 tokens.
- The loader now validates that all matched shards have the same format, sizes
  its batch buffer from the detected token width, and decodes both formats into
  the existing `int` input/target arrays.
- Verified with a host-only synthetic-shard smoke test plus
  `make train_gpt2cu` and `make train_llama3cu`.

## 2026-05 — M6 Llama eval loader dispatch

- Extended `EvalLoader` in [`llmc/dataloader.h`](llmc/dataloader.h) to detect
  HellaSwag eval format from the header: GPT-2 remains magic `20240522` v1 with
  uint16 records, and Llama-3 uses magic `20240802` v7 with uint32 records.
- The existing `inputs`/`targets`/`mask`/`label` API is unchanged; only the file
  parser and skip logic now account for token width and the wider Llama start
  delimiter.
- Verified with a host-only synthetic eval smoke test, `make all`, and
  `make train_llama3cu`.

## 2026-05 — M6 Llama MLP checkpoint layout fix

- Corrected the `train_llama3.cu` MLP parameter names to match
  `train_llama3.py::write_tensors`: Python `c_fc` / Meta `w3` is `fcw_up`, and
  Python `c_fc2` / Meta `w1` is `fcw_gate`.
- This prevents the C++ SwiGLU path from applying `silu()` to the wrong
  projection during checkpoint-backed training.

## 2026-05 — M6 Llama checkpoint size validation

- `train_llama3.cu` now validates `.bin` checkpoint payload size after parsing
  the `20240803` v5 header and parameter layout.
- A synthetic tiny BF16 checkpoint reaches the host-only dry-run path, while a
  two-byte-truncated copy fails with an explicit expected-vs-actual byte count.

## 2026-05 — M6 Llama HellaSwag preprocessing

- Extended [`dev/data/hellaswag.py`](dev/data/hellaswag.py) with
  `--model_desc {gpt-2,llama-3}`. GPT-2 keeps the existing
  `hellaswag_val.bin` output; Llama-3 writes `hellaswag_val_llama3.bin`.
- Extended [`dev/data/data_common.py`](dev/data/data_common.py) so eval files
  can be written in the existing GPT-2 uint16 format or a new Llama-3 uint32
  format with magic `20240802` v7.
- Verified with `python3 -m py_compile` and small local writer smoke tests for
  both GPT-2 and Llama-3 eval headers.

## 2026-05 — M6 TK RoPE wrapper

- Added [`llmc/tk/rope_tk.cuh`](llmc/tk/rope_tk.cuh), a raw-pointer fork of
  `ThunderKittens/kernels/rotary/rotary.cu` for bf16 RoPE over `(B,H,T,HS)`.
- Added [`llmc/rope.cuh`](llmc/rope.cuh), the C-style wrapper exposing
  `rope_forward` and `rope_backward`; backward uses the inverse rotation
  (`sin -> -sin`).
- Compile-checked with the H100 gencode path. Runtime numerical validation is
  still pending with the future `train_llama3.cu` integration.

## 2026-05 — M6 RMSNorm primitive

- Added [`llmc/tk/rmsnorm_tk.cuh`](llmc/tk/rmsnorm_tk.cuh), a TK forward and
  fused-residual forward fork mirroring `layernorm_tk` without mean subtraction
  or bias. Supported widths match the LayerNorm fork:
  `{768, 1024, 1280, 1600, 2048, 4096}`.
- Added [`llmc/rmsnorm.cuh`](llmc/rmsnorm.cuh), the C-style wrapper with CUDA
  fallback forward, fused-residual forward, and a plain CUDA backward
  correctness baseline for `dinp` and `dweight`.
- Compile-checked with the H100 gencode path. Runtime numerical validation and
  integration into `train_llama3.cu` remain pending.

## 2026-05 — M6/M7 Llama launch scripts

- Added [`scripts/run_llama3_1B.sh`](scripts/run_llama3_1B.sh), the 8xH100
  ZeRO-1 single-node Llama-3 1B target with B=32, T=2048, LR=3e-4,
  warmup=2000, cosine decay to 0.1, and the same H100 NCCL defaults as the GPT
  scripts.
- Added [`scripts/multi_node/run_llama3_8B_fs.sbatch`](scripts/multi_node/run_llama3_8B_fs.sbatch),
  the 2-node filesystem-rendezvous Llama-3 8B target for ZeRO-2.
- Both scripts syntax-check. Runtime execution waits for HF checkpoint
  availability, H100/NCCL access, TK GQA numerical validation, and RoPE fusion.

## 2026-05 — M6 Llama trainer surface and GQA baseline

- Added [`llmc/attention_gqa.cuh`](llmc/attention_gqa.cuh), a slow plain-CUDA
  GQA forward/backward correctness baseline. It permutes packed Llama Q/K/V,
  applies RoPE to Q/K, repeats KV logically across query groups, and recomputes
  softmax statistics in backward.
- Initially added [`llmc/tk/attention_gqa_h100.cuh`](llmc/tk/attention_gqa_h100.cuh)
  as the high-risk TK GQA kernel slot; the later GQA TK forward slice fills the
  forward path.
- Added [`train_llama3.cu`](train_llama3.cu), initially as a compile-ready
  Llama entrypoint surface with `LlamaConfig`, parameter layout, `llama3:1B` /
  `llama3:8B` / `llama3.1:8B` descriptor parsing, and `20240803` v5
  checkpoint-header parsing.
- `make train_llama3cu NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0` compiles.

## 2026-05 — M6 Llama trainer loop compile-wired

- Extended [`train_llama3.cu`](train_llama3.cu) from a dry entrypoint into a
  compile-wired trainer loop using the slow GQA correctness baseline:
  checkpoint/random initialization, RoPE-cache generation, Llama
  forward/backward/update, fused classifier loss, validation, Llama HellaSwag
  eval routing, AdamW, grad norm, ZeRO-0/1 gradient reduction hooks, and initial
  model-only checkpoint output.
- Added deterministic bucketed token-embedding gradient accumulation for Llama
  WTE gradients, mirroring the GPT path without position embeddings.
- The default `-x 0` path remains a host-only dry run for descriptor parsing and
  checkpoint payload-size validation. Training (`-x >0`) still needs H100
  runtime validation, TK GQA numerical validation, and the remaining RoPE-fusion
  work.

## 2026-05 — M6 GQA TK forward slice

- Replaced the GQA TK placeholder with a causal H100 BF16 forward wrapper in
  [`llmc/tk/attention_gqa_h100.cuh`](llmc/tk/attention_gqa_h100.cuh). It adapts
  the MHA template for grouped-query attention by launching over query heads and
  mapping each query head to its shared KV head with `n_rep = n_q / n_kv`.
- Updated [`llmc/attention_gqa.cuh`](llmc/attention_gqa.cuh) to dispatch the TK
  forward path for supported shapes, convert TK's `(B, NH, T, HS)` output back
  to the trainer's `(B, T, NH, HS)` layout, and fall back to the slow CUDA
  baseline for unsupported shapes.
- `train_llama3.cu` passes its existing per-layer output buffer as temporary TK
  forward workspace. RoPE fusion and runtime validation remain pending.
- Verification: `make train_llama3cu NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0`
  compiles. Runtime validation remains blocked by the local CUDA driver/runtime
  mismatch.

## 2026-05 — M6 GQA reference smoke target

- Added [`dev/cuda/test_attention_gqa.cu`](dev/cuda/test_attention_gqa.cu), a
  self-contained GQA + RoPE smoke harness for B=1, T=128, head_dim=128. It
  compares wrapper forward output and packed backward gradients against an
  independent CPU reference for packed Llama Q/K/V, RoPE rotation, causal GQA
  softmax, and inverse-RoPE gradient packing.
- Added `make test_attention_gqa` to the top-level [`Makefile`](Makefile).
- Verification: `make test_attention_gqa NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0`
  compiles. Runtime execution is blocked locally by the CUDA driver/runtime
  mismatch.

## 2026-05 — M6 GQA TK backward compile wiring

- Generalized [`llmc/tk/attention_h100.cuh`](llmc/tk/attention_h100.cuh)'s TK
  MHA backward launcher to accept separate query-head and KV-head counts, so
  grouped-query attention can reuse the existing H100 backward kernel with
  `hr = n_q_heads / n_kv_heads`.
- Added supported-shape TK GQA backward dispatch in
  [`llmc/tk/attention_gqa_h100.cuh`](llmc/tk/attention_gqa_h100.cuh) and
  [`llmc/attention_gqa.cuh`](llmc/attention_gqa.cuh). Unsupported shapes still
  fall back to the slow CUDA recompute baseline, and RoPE inverse remains a
  separate wrapper call after attention backward.
- Added Llama activation workspaces in [`train_llama3.cu`](train_llama3.cu) for
  permuted output/doutput BF16 scratch and TK backward FP32 `d`, `qg`, `kg`,
  and `vg` buffers.
- Extended [`dev/cuda/test_attention_gqa.cu`](dev/cuda/test_attention_gqa.cu)
  with a `T=256` case that passes the extra workspaces and exercises the
  supported-shape TK backward path on H100.
- Verification: `make train_llama3cu NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0`,
  `make test_attention_gqa NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0`, and
  `make all NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0` compile. Runtime
  execution remains blocked locally by the CUDA driver/runtime mismatch.

## 2026-05 — M6 RoPE/RMSNorm smoke targets

- Added [`dev/cuda/test_rope.cu`](dev/cuda/test_rope.cu), a CPU-reference smoke
  harness for RoPE forward and inverse-rotation backward over HS=64 and HS=128.
- Added [`dev/cuda/test_rmsnorm.cu`](dev/cuda/test_rmsnorm.cu), a CPU-reference
  smoke harness for RMSNorm forward, fused-residual forward, saved `rstd`,
  `dinp`, and `dweight`.
- Added [`dev/cuda/test_swiglu.cu`](dev/cuda/test_swiglu.cu), a CPU-reference
  smoke harness for SwiGLU forward, `dgate`, and `dup`.
- Added `make test_rope`, `make test_rmsnorm`, and `make test_swiglu` to the top-level
  [`Makefile`](Makefile).
- Verification: `make test_rope NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0`
  `make test_rmsnorm NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0`, and
  `make test_swiglu NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0` compile.
  Runtime execution remains blocked locally by the CUDA driver/runtime
  mismatch.

## 2026-05 — GPT-2 compile path and correctness baselines (M2/M3 partial)

- Added [`llmc/tk/attention_h100.cuh`](llmc/tk/attention_h100.cuh), copied from
  ThunderKittens' H100 MHA forward kernel, and [`llmc/attention.cuh`](llmc/attention.cuh)
  with the llm.c QKV permute/unpermute glue. The TK kernel requires
  `T % 192 == 0`; the wrapper pads non-aligned sequence lengths into scratch,
  runs TK at `Tpad`, and unpads back to the normal output layout.
- Added [`llmc/layernorm.cuh`](llmc/layernorm.cuh) from llm.c as the LayerNorm /
  fused-residual correctness baseline, then added
  [`llmc/tk/layernorm_tk.cuh`](llmc/tk/layernorm_tk.cuh) as the TK forward fork.
  Forward and fused-residual forward now route through TK for supported widths
  `{768, 1024, 1280, 1600, 2048, 4096}`; backward remains the llm.c CUDA
  baseline until the M3 TK primitive rewrite lands.
- Added [`train_gpt2.cu`](train_gpt2.cu), ported from llm.c: cuBLAS/cuDNN paths
  stripped, local wrapper calls wired, GELU fusion split into explicit
  `matmul_forward` + `gelu_forward`, and 128-byte parameter-offset assertions
  added for TK TMA alignment.
- Replaced the runtime stubs in `matmul_backward` and `attention_backward` with
  slow plain-CUDA correctness baselines. These are not the target M3 TK kernels;
  they exist to make the trainer/test compile path complete while the TK
  transposed GEMM and MHA backward ports are still pending.
- Added [`test_gpt2.cu`](test_gpt2.cu), ported from llm.c, and updated `make all`
  to build `test_matmul`, `train_gpt2cu`, and `test_gpt2cu`.
- Renamed the AdamW helper `lerp` to avoid a C++20 ambiguity with `std::lerp`.

Verification:

- `make all NO_MULTI_GPU=1 NO_USE_MPI=1 FORCE_NVCC_O=0` compiles
  `test_matmul`, `train_gpt2cu`, and `test_gpt2cu`.
- Runtime validation was not run: this sandbox cannot access the GPU
  (`nvidia-smi` reports GPU access blocked), and the GPT-2 starter-pack `.bin`
  files are not present locally.

## 2026-05 — documentation pass (M8 in flight)

- Added [`goal.md`](goal.md) at the repo root as the single-source-of-truth
  TODO list. Per-milestone, per-task checkboxes; lists upstream reference
  files for every pending wrapper. Why: the project is mid-port and contributors
  (human and LLM) need an unambiguous view of the surface.
- Added [`docs/`](docs/) tree:
  [`docs/README.md`](docs/README.md) (index), [`docs/architecture.md`](docs/architecture.md),
  [`docs/build-and-run.md`](docs/build-and-run.md), [`docs/kernel-reference.md`](docs/kernel-reference.md),
  [`docs/precision.md`](docs/precision.md), [`docs/multi-gpu.md`](docs/multi-gpu.md),
  [`docs/llama3.md`](docs/llama3.md), [`docs/testing.md`](docs/testing.md),
  [`docs/porting-notes.md`](docs/porting-notes.md), [`docs/agents.md`](docs/agents.md).
  All grounded in the current source; status flags (✅/🟡/⬜) explicit per kernel.
- Added LLM ingestion artifacts: [`llms.txt`](llms.txt) (concise index) and
  [`llms-full.txt`](llms-full.txt) (full-tree bundle). Both linked from the
  README.
- Refreshed [`README.md`](README.md): replaces the launch-style status block
  with a docs map + status snapshot, adds a layout legend showing which files
  are pending per milestone, calls out the BF16 / sm_90a constraints up front.
- Added repo-local agent skill: [`.claude/skills/llm-kittens-port/`](.claude/skills/llm-kittens-port/).
  Routes future LLM agents into `goal.md` and the wrapper-PR checklist; does
  not duplicate the docs.

Verification: docs and source were cross-checked file-by-file; every kernel
status flag matches what is actually in the tree (and what is missing).

## 2026-05 — partial M2: TK GEMM wrapper

- Added [`llmc/tk/gemm_h100.cuh`](llmc/tk/gemm_h100.cuh): ThunderKittens bf16
  H100 GEMM ported into header form. Two specialisations exposed:
  `matmul_default<2,4,8>` for `N % 256 == 0` and `matmul_small_n<2,2,8>` for
  `N % 128 == 0`. Persistent grid (132 SMs, the H100 SM count); TMA producer;
  WGMMA consumer.
- Added [`llmc/tk/tk_common.cuh`](llmc/tk/tk_common.cuh): the bridge layer.
  Hard `static_assert` on `floatX == __nv_bfloat16`; hard `#error` if
  `KITTENS_SM90` is not defined. Exposes `llmk::TK_ALIGN = 128` (TMA-aligned
  allocator constant) and `llmk::tk_set_max_dynamic_smem(...)`.
- Added [`llmc/matmul.cuh`](llmc/matmul.cuh): C-style `matmul_forward`
  dispatching between the two GEMM specialisations based on `OC % 256`. Bias
  is applied as a separate `add_bias_kernel` pass (cuBLASLt epilogue fusion
  intentionally dropped in v1, ~5% throughput cost). `matmul_backward_bias_kernel9`
  and `reduce_add_sum_kernel` ported verbatim from `llm.c/llmc/matmul.cuh:17,83`.
  `matmul_backward` was initially left as an M3 stub; it now has a slow
  correctness baseline, with the target TK implementation still pending.
- Added [`dev/cuda/test_matmul.cu`](dev/cuda/test_matmul.cu) and
  `make test_matmul`. Sweeps three shapes (1024³ square, GPT-2 124M MLP up,
  GPT-2 124M LM head) and compares against a naive bf16 reference with FP32
  accumulation. Tolerance 0.5 (well above the ~0.08 expected accumulation
  error for K=768 and bf16).

Why: M2 is the forward-path milestone. GEMM is the heaviest operator in the
graph; getting the wrapper + smoke test in early de-risks the rest of M2.

## 2026-05 — M1: skeleton, Makefile, verbatim ports

- Added [`Makefile`](Makefile) modelled on `llm.c`'s with the required
  ThunderKittens-specific changes:
  - `-arch=sm_90a` (the `a` suffix is required for WGMMA / TMA — `sm_90` alone
    is rejected by nvcc)
  - `-std=c++20` (TK requires it; llm.c uses C++17)
  - Default `TK_ROOT=$(abspath ../ThunderKittens)`
  - `-DENABLE_BF16`, `-DKITTENS_SM90`
  - cuBLAS, cuBLASLt, cuDNN dropped entirely
  - GPU-capability sniff with a warning if not Hopper
  - NCCL and MPI sniffed at configure time (set `NO_MULTI_GPU=1` /
    `NO_USE_MPI=1` to force-disable)
- Added [`llmc/cuda_common.h`](llmc/cuda_common.h): `floatX = __nv_bfloat16`
  locked. Compile-time `#error` if `ENABLE_FP16` or `ENABLE_FP32` is defined.
- Added verbatim ports of the element-wise / non-tile kernels and utilities
  from `llm.c/llmc/`:
  - [`encoder.cuh`](llmc/encoder.cuh), [`gelu.cuh`](llmc/gelu.cuh),
    [`fused_classifier.cuh`](llmc/fused_classifier.cuh),
    [`adamw.cuh`](llmc/adamw.cuh), [`global_norm.cuh`](llmc/global_norm.cuh),
    [`zero.cuh`](llmc/zero.cuh) (NCCL + ZeRO-0/1 + MPI / TCP / FS init),
    [`cuda_utils.cuh`](llmc/cuda_utils.cuh) (`x128`, `f128`, `stochastic_rounding`),
    [`dataloader.h`](llmc/dataloader.h), [`tokenizer.h`](llmc/tokenizer.h),
    [`sampler.h`](llmc/sampler.h), [`schedulers.h`](llmc/schedulers.h),
    [`rand.h`](llmc/rand.h), [`mfu.h`](llmc/mfu.h),
    [`outlier_detector.h`](llmc/outlier_detector.h), [`logger.h`](llmc/logger.h),
    [`utils.h`](llmc/utils.h).
  - [`cublas_common.h`](llmc/cublas_common.h) is kept as a stub for
    symbol-name compatibility — no cuBLAS / cuBLASLt symbol is referenced in v1.
- Added [`dev/data/`](dev/data/): full mirror of `llm.c/dev/data/`
  (`tinyshakespeare.py`, `tinystories.py`, `fineweb.py`, `fineweb.sh`,
  `edu_fineweb.sh`, `hellaswag.py`, `mmlu.py`, `data_common.py`, `README.md`).
  Both `gpt-2` and `llama-3` model descriptors are supported in the prep
  scripts; `dataloader.h` dispatches training shards on header magic at load
  time.
- Added [`dev/download_starter_pack.sh`](dev/download_starter_pack.sh): fetches
  `gpt2_tokenizer.bin`, `gpt2_124M.bin`, `gpt2_124M_bf16.bin`,
  `gpt2_124M_debug_state.bin` from Karpathy's HF mirror.
- Added [`profile_gpt2cu.py`](profile_gpt2cu.py): started from llm.c's
  nsight-compute post-processing helper and later adapted for this repo's
  TK-only build. The `profile_gpt2.cu` target it processes is M8.
- Added [`requirements.txt`](requirements.txt): `tqdm`, `numpy<2`, `torch`,
  `tiktoken`, `transformers`, `datasets`, `requests`. Used only by `dev/data/*.py`.

Why: M1 is the non-negotiable foundation. Dropping cuBLAS / cuDNN forces every
subsequent kernel through a TK or verbatim-CUDA path; locking BF16 forces
the design to commit to TK's precision constraint up front rather than
discovering it at the bottom of M2 or M3.
