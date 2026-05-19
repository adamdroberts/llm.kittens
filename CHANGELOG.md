# Changelog

Append-only history of meaningful changes to llm.kittens. Roughly grouped by
milestone. Adds within a milestone are listed in chronological order.

The canonical "what is done / what is left" is [`goal.md`](goal.md). The
changelog is the diary; `goal.md` is the plan.

## 2026-05-19 — SM120 RTX 5090 pure-TK optimization rounds

- Rejected scoped fused dInput+dGELU swizzle
  `LLMK_SM120_DINP_DGELU_SUPER_M=9`. The macro build passed `test_attention`
  and `test_matmul` (`10/10`). The focused benchmark showed the new
  FC-projection dInput+dGELU row slightly ahead of cuBLASLt in that run
  (`1814.95 us` versus `1824.86 us`), but this was slower than the
  no-override default diagnostic row measured in the previous round
  (`1794.23 us`). TinyStories 3-step validation averaged `2644.49 ms` with
  steps `2640.13`, `2641.68`, and `2647.31 ms`, slower than the current
  pure-TK rebaseline, so fused dGELU aliases keep the default inherited
  `SUPER_M=8`.
- Added a focused `bench_sm120_matmul` row for the FC-projection backward
  fused dInput+dGELU path and introduced
  `LLMK_SM120_DINP_DGELU_SUPER_M` as a default-preserving hook for fused dGELU
  dInput aliases. The default hook inherits `LLMK_SM120_DINP_SUPER_M=8`, so
  source-default behavior is unchanged unless a test build overrides it.
  Rejected `LLMK_SM120_DINP_DGELU_SUPER_M=7`: the macro build passed
  `test_attention` and `test_matmul` (`10/10`), but the new benchmark row
  measured FC-projection dInput+dGELU at `1818.06 us` versus cuBLASLt
  `1800.93 us`, and TinyStories 3-step validation averaged `2644.83 ms` with
  steps `2638.32`, `2642.12`, and `2647.55 ms`. The no-override source build
  passed `test_attention`; the first `test_matmul` hit the known intermittent
  MLP-up row and the rerun passed `10/10`. Its diagnostic benchmark measured
  the default FC-projection dInput+dGELU row at `1794.23 us` versus cuBLASLt
  `1882.68 us` in that run, so the hook stays available for scoped retests
  but the source default remains `SUPER_M=8`.
- Rejected removing the runtime `bias_present` guard from the SM120 NT
  forward bias epilogue. The source candidate passed `test_attention` and
  `test_matmul` (`10/10`). The focused benchmark was mixed: attention-projection
  forward won in that noisy run (`376.31 us` versus cuBLASLt `396.95 us`), but
  qkv forward still trailed (`1070.75 us` versus `1040.88 us`), FC fused
  forward stayed behind (`1537.55 us` versus `1493.35 us`), FC-projection
  forward stayed behind (`1418.96 us` versus `1388.13 us`), and LM-head forward
  was unchanged because it has no bias. TinyStories 3-step validation averaged
  `2645.17 ms` with steps `2639.19`, `2640.39`, and `2649.95 ms`, slower than
  the current pure-TK rebaseline, so the epilogue keeps the runtime guard.
- Rejected a temporary qkv-forward-only swizzle route with
  `LLMK_SM120_QKV_FORWARD_SUPER_M=6`. The hook routed only regular qkv
  forward GEMMs (`N == 3*K`) through a dedicated `matmul_qkv_nt_bias` alias
  while leaving FC fused, projection, and huge-N forward aliases unchanged. The
  source candidate passed `test_attention` and `test_matmul` (`10/10`), but the
  focused benchmark regressed the target row: qkv forward was `1152.60 us`
  versus cuBLASLt `1062.89 us`. TinyStories 3-step validation averaged
  `2645.98 ms` with steps `2639.88`, `2642.41`, and `2649.55 ms`, slower than
  the current pure-TK rebaseline, so the temporary alias and dispatch hook were
  removed.
- Rejected a temporary huge-N `256x192x32` forward tile selector,
  `LLMK_SM120_HUGE_N_N192=1`, for LM-head-style SM120 forward GEMMs. The
  source candidate passed `test_attention` and `test_matmul` (`10/10`). The
  macro benchmark slightly improved LM-head forward to `24550.62 us`, and the
  no-override promotion check measured `24516.15 us`, but both still missed
  cuBLASLt (`22201.70 us` and `22295.02 us` respectively). The candidate did
  not improve end-to-end training: macro TinyStories 3-step validation
  averaged `2622.19 ms` with steps `2616.01`, `2618.70`, and `2625.68 ms`,
  while the no-override default check averaged `2623.57 ms` with steps
  `2616.12`, `2620.90`, and `2626.23 ms`. Since the trainer path did not beat
  the current pure-TK rebaseline, the temporary tile selector was removed and
  huge-N forward remains on the `256x128x32` tile.
- Rejected a temporary fused-forward-only swizzle hook,
  `LLMK_SM120_FORWARD_GELU_SUPER_M=8`, scoped to the `*_nt_bias_gelu` aliases
  while leaving plain forward, dInput, and dWeight aliases on their accepted
  defaults. The source candidate passed `test_attention` and `test_matmul`
  (`10/10`). The focused benchmark did not close the targeted FC fused-forward
  gap: FC fused forward was `1534.92 us` versus cuBLASLt `1471.32 us`, while
  qkv forward stayed `1072.70 us` versus `1038.62 us`, FC-projection forward
  stayed `1465.95 us` versus `1342.56 us`, and LM-head forward stayed
  `24738.24 us` versus `22115.12 us`. TinyStories 3-step validation averaged
  `2626.14 ms` with steps `2618.47`, `2622.93`, and `2629.35 ms`, slower than
  the current pure-TK rebaseline, so the temporary hook was removed.
- Rejected non-direct dInput swizzle `LLMK_SM120_DINP_SUPER_M=1`. The macro
  build passed `test_attention` and `test_matmul` (`10/10`). The focused
  benchmark improved qkv dInput in that run (`1019.07 us` versus cuBLASLt
  `1076.31 us`) but still missed the broader target: FC dInput was
  `1408.10 us` versus `1341.31 us`, FC-projection dInput was `1477.03 us`
  versus `1416.01 us`, LM-head dInput was `22170.99 us` versus `21627.76 us`,
  and forward plus several dWeight rows remained behind cuBLASLt. TinyStories
  3-step validation averaged `2625.47 ms` with steps `2619.91`, `2622.00`,
  and `2628.95 ms`, slower than the current pure-TK rebaseline, so dInput
  keeps `LLMK_SM120_DINP_SUPER_M=8`.
- Rejected a temporary large-K-only direct-Bcol dInput swizzle split with
  `LLMK_SM120_DINP_DIRECT_BCOL_LARGEK_SUPER_M=7`. The hook routed only
  LM-head-style large-K direct-Bcol dInput through a separate alias while
  leaving small-K dInput on the accepted direct-Bcol path. The build passed
  `test_attention`; the first `test_matmul` hit the known intermittent MLP-up
  forward row, and the immediate rerun passed `10/10`. The focused benchmark
  did not close the targeted LM-head dInput gap (`22267.63 us` versus cuBLASLt
  `21607.19 us`) and still left forward plus several dWeight rows behind
  cuBLASLt. TinyStories 3-step validation averaged `2627.00 ms` with steps
  `2620.80`, `2623.67`, and `2630.34 ms`, slower than the current pure-TK
  rebaseline, so the temporary alias and dispatch split were removed.
- Validated the corrected selective cuBLASLt fallback diagnostic for forward
  and dInput, built with `LLMK_SM120_CUBLASLT_FORWARD_FALLBACK=1` and
  `LLMK_SM120_CUBLASLT_DINP_FALLBACK=1` while leaving dWeight on the SM120 TK
  path. The corrected link command passed cuBLASLt explicitly through
  `NVCC_LDLIBS` so it did not define full `LLMK_SM120_USE_CUBLASLT_GEMM`. The
  build passed `test_attention` and `test_matmul` (`10/10`). TinyStories
  3-step validation averaged `2527.49 ms` with steps `2520.85`, `2524.51`,
  and `2530.48 ms`, slightly faster than the full cuBLASLt fallback selector
  (`2532.28 ms`). This narrows the remaining TK-side work: forward and dInput
  are the high-leverage gaps, while keeping current TK dWeight was not the
  limiter in this fallback mix.
- Rejected a temporary forward-only swizzle hook,
  `LLMK_SM120_FORWARD_SUPER_M=5`, scoped to NT forward aliases while preserving
  the accepted dInput and dWeight swizzles. The source candidate passed
  `test_attention` and `test_matmul` (`10/10`). The focused benchmark did not
  close the CUDA target: qkv forward was `1073.94 us` versus cuBLASLt
  `1041.26 us`, FC fused forward was `1587.64 us` versus `1472.77 us`,
  FC-projection forward was `1466.91 us` versus `1343.42 us`, and LM-head
  forward was `24962.38 us` versus `22101.15 us`. TinyStories 3-step
  validation averaged `2628.15 ms` with steps `2620.61`, `2625.19`, and
  `2631.12 ms`, slower than the current pure-TK rebaseline, so the temporary
  hook was removed.
- Rejected pure SM120 TK codegen with
  `-Xptxas=--allow-expensive-optimizations=false`. The build passed
  `test_attention` and `test_matmul` (`10/10`). The focused benchmark showed
  the disabled ptxas pass is not a useful tradeoff: qkv forward regressed to
  `1123.54 us` versus cuBLASLt `1090.06 us`, attention-projection dInput
  regressed to `378.94 us` versus `367.83 us`, FC-projection dInput remained
  `1476.43 us` versus `1365.09 us`, and LM-head forward remained
  `24991.15 us` versus `22097.84 us`. TinyStories 3-step validation averaged
  `2625.11 ms` with steps `2618.85`, `2621.20`, and `2629.01 ms`, slower than
  the current pure-TK rebaseline, so pure SM120 TK keeps the existing ptxas
  optimization policy.
- Rejected a temporary forward-only swizzle hook,
  `LLMK_SM120_FORWARD_SUPER_M=6`, that kept backward aliases on the accepted
  swizzles while retesting a scoped variant of the previously rejected global
  `SUPER_M=6` setting. The source candidate passed `test_attention` and
  `test_matmul` (`10/10`). The focused benchmark still missed the broad target:
  qkv forward was ahead (`1073.78 us` versus cuBLASLt `1088.53 us`), but FC
  fused forward was `1582.01 us` versus `1470.55 us`, FC-projection forward
  was `1469.53 us` versus `1343.15 us`, LM-head forward was `24756.56 us`
  versus `22209.70 us`, and several dInput/dWeight rows remained slower than
  cuBLASLt. TinyStories 3-step validation averaged `2626.06 ms` with steps
  `2620.08`, `2621.40`, and `2630.73 ms`, slower than the current pure-TK
  rebaseline, so the temporary hook was removed.
- Rejected a temporary forward-only swizzle hook,
  `LLMK_SM120_FORWARD_SUPER_M=4`, that routed all NT forward aliases through a
  separate swizzle while leaving dInput and dWeight aliases on their existing
  defaults. The source candidate passed `test_attention` and `test_matmul`
  (`10/10`). The focused benchmark had an isolated qkv forward win
  (`1073.61 us` versus cuBLASLt `1087.91 us`), but FC fused forward regressed
  to `1582.42 us` versus `1514.37 us`, FC-projection forward stayed behind
  (`1488.42 us` versus `1343.01 us`), and LM-head forward stayed far behind
  (`25128.29 us` versus `22189.73 us`). TinyStories 3-step validation
  averaged `2628.47 ms` with steps `2622.42`, `2626.28`, and `2630.65 ms`,
  slower than the current pure-TK rebaseline, so the temporary hook was
  removed and forward aliases continue to inherit `LLMK_SM120_SUPER_M=7`.
- Rejected the corrected selective cuBLASLt fallback diagnostic for forward and
  dWeight, built with `LLMK_SM120_CUBLASLT_FORWARD_FALLBACK=1` and
  `LLMK_SM120_CUBLASLT_DWEIGHT_FALLBACK=1` while leaving dInput on the SM120 TK
  path. The corrected link command passed cuBLASLt explicitly through
  `NVCC_LDLIBS` so it did not define full `LLMK_SM120_USE_CUBLASLT_GEMM`. The
  build passed `test_attention` and `test_matmul` (`10/10`). TinyStories
  3-step validation averaged `2577.96 ms` with steps `2573.94`, `2575.75`,
  and `2580.18 ms`: faster than pure TK and the supplied llm.c baseline, but
  slower than the forward-only fallback diagnostic (`2564.07 ms`) and the full
  cuBLASLt fallback selector (`2532.28 ms`). This confirms the forward path is
  still the highest-leverage TK target; adding dWeight fallback did not improve
  the current selector mix.
- Rejected direct-Bcol dInput swizzle
  `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=15`. The macro build passed
  `test_attention` and `test_matmul` (`10/10`). The focused benchmark kept qkv
  dInput competitive (`1019.40 us` versus cuBLASLt `1052.73 us`) but regressed
  attention-projection dInput to `378.92 us` versus `367.94 us`, left
  FC/FC-projection/LM-head dInput behind cuBLASLt, and did not close forward or
  dWeight gaps. TinyStories 3-step validation averaged `2626.81 ms` with steps
  `2620.20`, `2623.29`, and `2630.34 ms`, slower than the current pure-TK
  rebaseline, so direct-Bcol dInput keeps `SUPER_M=8`.
- Rejected dWeight swizzle `LLMK_SM120_DWEIGHT_SUPER_M=20`. The macro build
  passed `test_attention` and `test_matmul` (`10/10`). The focused benchmark
  improved qkv dWeight relative to several rejected high-swizzle probes but
  still missed the CUDA target (`1201.54 us` versus cuBLASLt `1156.49 us`) and
  regressed other material dWeight rows, including attention-projection dW+=
  (`509.48 us` versus `377.97 us`), FC dWeight (`1597.88 us` versus
  `1490.55 us`), and FC-projection dW+= (`1612.78 us` versus `1470.33 us`).
  TinyStories 3-step validation averaged `2628.27 ms` with steps `2623.31`,
  `2624.36`, and `2632.19 ms`, slower than the current pure-TK rebaseline, so
  dWeight keeps `LLMK_SM120_DWEIGHT_SUPER_M=2`.
- Rejected a temporary huge-N-forward-only swizzle hook with
  `LLMK_SM120_HUGE_N_FORWARD_SUPER_M=6`. The hook isolated LM-head-style
  forward GEMMs from the shared `LLMK_SM120_SUPER_M=7` swizzle and left
  ordinary forward, dInput, and dWeight aliases unchanged. The macro build
  passed `test_attention`; the first `test_matmul` hit the known intermittent
  MLP-up forward row, and an immediate rerun passed `10/10`. The focused
  benchmark did not close the target: LM-head forward stayed `24981.70 us`
  versus cuBLASLt `22272.49 us`, FC fused forward regressed to `1607.62 us`,
  and material dWeight rows still trailed. TinyStories 3-step validation
  averaged `2624.94 ms` with steps `2619.93`, `2621.55`, and `2628.34 ms`,
  slower than the current pure-TK rebaseline, so the temporary hook was
  removed and huge-N forward continues to inherit `LLMK_SM120_SUPER_M=7`.
- Validated a corrected forward-only cuBLASLt fallback diagnostic with
  `LLMK_SM120_CUBLASLT_FORWARD_FALLBACK=1`, explicit cuBLASLt linkage through
  `NVCC_LDLIBS`, and dInput/dWeight left on the SM120 TK paths. The build
  passed `test_attention` and `test_matmul` (`10/10`). TinyStories 3-step
  validation averaged `2564.07 ms` with steps `2558.60`, `2561.36`, and
  `2566.79 ms`, much closer to the full cuBLASLt fallback selector
  (`2532.28 ms`) than the pure-TK rebaseline (`2621.86 ms`). This points the
  next kernel work at forward/large-N LM-head throughput rather than
  backward-only GEMM replacement.
- Rejected CUDA bias-gradient block size `LLMK_SM120_BIAS_BLOCK_SIZE=704`.
  This fills the gap between previously rejected 640- and 768-thread bias
  reduction launches. The macro build passed `test_attention` and
  `test_matmul` (`10/10`), but TinyStories 3-step validation averaged
  `2626.37 ms` with steps `2620.43`, `2622.99`, and `2629.74 ms`, slower than
  the current pure-TK rebaseline, so the source keeps the 512-thread bias
  block.
- Rejected direct-Bcol dInput swizzle `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=14`
  at the matmul smoke gate. The macro build passed `test_attention`, but
  `test_matmul` failed the GPT-2 MLP-up forward row twice (`5.2422` then
  `6.1094` max diff versus the `0.50` tolerance). No focused benchmark or
  TinyStories 3-step validation was run for the invalid candidate, and the
  source keeps direct-Bcol dInput `SUPER_M=8`.
- Rejected shared SM120 grid swizzle `LLMK_SM120_SUPER_M=14`. The macro build
  passed `test_attention` and `test_matmul` (`10/10`), but the focused
  benchmark stayed short of the per-kernel target: qkv forward was
  `1072.82 us` versus cuBLASLt `1038.93 us`, FC-projection forward worsened to
  `1517.15 us` versus `1346.06 us`, LM-head forward remained `25008.97 us`
  versus `22186.15 us`, and qkv/attention-projection/LM-head dWeight rows
  still trailed. TinyStories 3-step validation averaged `2621.47 ms` with
  steps `2617.38`, `2617.86`, and `2625.09 ms`, effectively tied with the
  current pure-TK rebaseline and still behind the full cuBLASLt fallback
  selector, so the source keeps shared `LLMK_SM120_SUPER_M=7`.
- Rejected the corrected selective cuBLASLt fallback diagnostic for dInput and
  dWeight only, built with
  `LLMK_SM120_CUBLASLT_DINP_FALLBACK=1` and
  `LLMK_SM120_CUBLASLT_DWEIGHT_FALLBACK=1` while leaving forward GEMMs on the
  SM120 path. The corrected link command passed cuBLASLt explicitly through
  `NVCC_LDLIBS` so it did not define full `LLMK_SM120_USE_CUBLASLT_GEMM`. The
  build passed `test_attention` and `test_matmul` (`10/10`). TinyStories
  3-step validation averaged `2612.23 ms` with steps `2614.42`, `2609.52`,
  and `2614.93 ms`: faster than the current pure-TK rebaseline but still well
  behind the full cuBLASLt fallback selector (`2532.28 ms`). This confirms the
  remaining optimization target is not solved by only replacing backward GEMMs;
  forward and LM-head-side SM120 kernels still need work.
- Rejected disabling the small-K SM120 direct-Bcol dInput route with
  `LLMK_SM120_DINP_DIRECT_BCOL_SMALLK=0`. The macro build passed
  `test_attention`; the first `test_matmul` hit the known intermittent MLP-up
  forward row, and an immediate rerun passed `10/10`. The focused benchmark
  lost the attention-projection dInput win (`380.52 us` versus cuBLASLt
  `370.80 us`) while qkv dInput (`1087.51 us` versus `1034.90 us`), FC dInput
  (`1450.41 us` versus `1387.80 us`), FC-projection dInput (`1476.76 us`
  versus `1419.44 us`), and LM-head dInput (`22345.27 us` versus `21689.33 us`)
  all still trailed. TinyStories 3-step validation averaged `2645.32 ms` with
  steps `2640.03`, `2641.98`, and `2648.66 ms`, slower than the current
  pure-TK rebaseline, so the small-K direct-Bcol route remains enabled.
- Rejected lowering dWeight split-K to `LLMK_SM120_DWEIGHT_SPLIT_K=2`. The
  macro build passed `test_attention`; the first `test_matmul` hit the known
  intermittent MLP-up forward row, and an immediate rerun passed `10/10`. The
  focused benchmark showed why the lower fanout is not viable: qkv dWeight
  regressed to `1567.93 us` versus cuBLASLt `1102.18 us`, and
  attention-projection dWeight regressed to `1132.30 us` versus `372.10 us`.
  TinyStories 3-step validation averaged `2688.58 ms` with steps `2683.31`,
  `2683.68`, and `2693.47 ms`, much slower than the current pure-TK
  rebaseline, so dWeight split-K stays at `8`.
- Rejected direct-Bcol dInput `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=13`. The
  macro build passed `test_attention` and `test_matmul` (`10/10`), and the
  focused benchmark showed isolated wins for attention-projection forward
  (`376.68 us` versus cuBLASLt `414.93 us`) and attention-projection dInput
  (`354.92 us` versus `367.79 us`). The broad target still failed: qkv dInput
  was `1066.70 us` versus `1009.68 us`, FC dInput was `1359.98 us` versus
  `1340.46 us`, FC-projection dInput stayed `1527.13 us` versus `1367.71 us`,
  and dWeight/LM-head rows remained behind. TinyStories 3-step validation
  averaged `2625.26 ms` with steps `2619.78`, `2622.30`, and `2628.21 ms`,
  slower than the current pure-TK rebaseline, so direct-Bcol dInput keeps
  `SUPER_M=8`.
- Rejected narrowing the SM120 small-K direct-Bcol dInput route to
  `LLMK_SM120_DINP_DIRECT_BCOL_K_CAP=1024`, which leaves only the
  attention-projection dInput shape on the direct B-column path and routes qkv
  plus FC/MLP-up dInput back through the row-load path. The macro build passed
  `test_attention` and `test_matmul` (`10/10`), but the focused benchmark
  stayed mixed: attention-projection dInput improved to `356.10 us` versus
  cuBLASLt `365.10 us`, while qkv dInput regressed to `1089.25 us` versus
  `1006.10 us`, FC dInput still trailed (`1448.56 us` versus `1390.07 us`),
  and dWeight/LM-head rows remained behind. TinyStories
  3-step validation averaged `2642.59 ms` with steps `2638.00`, `2639.57`,
  and `2645.62 ms`, slower than the current pure-TK rebaseline, so the source
  keeps the broader accepted direct-Bcol cap.
- Rejected disabling TN dWeight direct B-column register loads with
  `LLMK_SM120_TN_DIRECT_B_COL=0`. The macro build passed `test_attention` and
  `test_matmul` (`10/10`), but the focused benchmark worsened the targeted
  dWeight rows: qkv dWeight was `1240.07 us` versus cuBLASLt `1152.99 us`,
  attention-projection dWeight was `516.51 us` versus `373.33 us`, FC-projection
  dWeight was `1512.77 us` versus `1469.90 us`, and LM-head dWeight regressed
  to `22933.20 us` versus `20837.07 us`. TinyStories 3-step validation
  averaged `2654.96 ms` with steps `2651.06`, `2652.11`, and `2657.81 ms`, so
  TN dWeight keeps the promoted direct B-column load path.
- Rejected a large-K-only direct-Bcol dInput swizzle split with
  `LLMK_SM120_DINP_DIRECT_BCOL_LARGEK_SUPER_M=4`. The temporary alias targeted
  only LM-head-style large-K direct-Bcol dInput and left small-K direct-Bcol
  GEMMs on the accepted `SUPER_M=8` route. The build passed `test_attention`;
  the first `test_matmul` run hit the known transient MLP-up forward row, and
  an immediate rerun passed `10/10`. The focused benchmark did not close the
  targeted LM-head dInput gap (`22252.03 us` versus cuBLASLt `21668.04 us`) and
  worsened attention-projection dInput (`406.75 us` versus `366.94 us`).
  TinyStories 3-step validation averaged `2622.16 ms` with steps `2617.65`,
  `2619.38`, and `2624.95 ms`, essentially tied but slower than the current
  pure-TK rebaseline, so the temporary large-K alias and dispatch hook were
  removed.
- Rejected the combined dWeight macro probe
  `LLMK_SM120_DWEIGHT_SUPER_M=1` with `LLMK_SM120_DWEIGHT_SPLIT_K=16`. The
  combination was meant to pair the lower dWeight swizzle with the qkv-only
  16-way split-K path. It compiled and `test_attention` passed all three smoke
  shapes, but `test_matmul` failed the GPT-2 MLP-up forward row twice
  (`8.0938` then `5.4609` max diff versus the `0.50` tolerance). No focused
  benchmark or TinyStories validation was run for the invalid candidate.
- Rejected a scoped qkv-only direct-Bcol dInput swizzle split. The temporary
  source route used `SUPER_M=3` only for the qkv-shaped direct-Bcol dInput GEMM
  (`N == 768`, `K == 2304`) and left attention-projection, FC, and LM-head on
  the accepted `SUPER_M=8` route. The build passed `test_attention` and
  `test_matmul` (`10/10`). The focused benchmark was mixed: qkv forward moved
  slightly ahead of cuBLASLt (`1073.40 us` versus `1089.31 us`) and
  attention-projection dInput was faster (`353.92 us` versus `367.87 us`), but
  qkv dInput still trailed (`1017.91 us` versus `1009.42 us`), material
  dWeight rows remained behind, and FC/FC-projection forward regressed.
  TinyStories 3-step validation averaged `2623.35 ms` with steps `2618.26`,
  `2620.64`, and `2626.05 ms`, slower than the current pure-TK rebaseline, so
  the temporary qkv alias and dispatch hook were removed.
- Rebaselined the current pure-TK SM120 source after promoting the faster
  cuBLASLt fallback selector. The pure-TK build passed `test_attention` (all
  three smoke shapes) and `test_matmul` (`10/10`). The focused benchmark now
  compares against the stronger cuBLASLt selector: FC dWeight was nearly tied
  (`1482.14 us` versus `1474.01 us`), FC-projection dWeight was ahead
  (`1453.56 us` versus `1530.79 us`), but qkv dWeight (`1221.61 us` versus
  `1104.13 us`), attention-projection dWeight (`466.73 us` versus
  `374.67 us`), FC-projection dInput (`1527.93 us` versus `1365.88 us`), and
  LM-head forward/dInput/dWeight remained behind. TinyStories 3-step validation
  averaged `2621.86 ms` with steps `2617.27`, `2618.90`, and `2624.82 ms`,
  still faster than the supplied llm.c baseline but short of the per-kernel
  CUDA-variant goal.
- Rejected the opposite SM120 cuBLASLt fallback selector,
  `LLMK_SM120_CUBLASLT_SELECT_MAX_WAVES=1`. The cached fallback build passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes), but
  TinyStories 3-step validation regressed to `3319.89 ms` with steps
  `3314.09`, `3315.91`, and `3323.87 ms`. The promoted fallback default
  therefore remains the lower-wave selector.
- Promoted the faster SM120 cuBLASLt fallback heuristic policy. The fallback
  helper now requests 8 cuBLASLt heuristic results by default and, unless an
  explicit heuristic index or max-wave selector is provided, chooses the
  lowest-wave algorithm. The normal `SM120_USE_CUBLASLT_GEMM=1` build passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes).
  TinyStories 3-step validation averaged `2532.28 ms` with steps `2534.57`,
  `2530.16`, and `2534.40 ms`, slightly faster than the explicit
  `LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS=8` diagnostic and about `10.15%`
  faster than the supplied llm.c baseline average (`2818.23 ms`). This improves
  the CUDA fallback ceiling only; pure-TK dWeight and LM-head kernels remain
  the optimization target.
- Validated a current full SM120 cuBLASLt fallback diagnostic with cached plans
  and wider heuristic selection:
  `LLMK_SM120_USE_CUBLASLT_GEMM=1`, `LLMK_SM120_CACHE_CUBLASLT_PLANS=1`,
  `LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS=8`, and
  `LLMK_SM120_CUBLASLT_SELECT_MIN_WAVES=1`. The build passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes).
  TinyStories 3-step validation averaged `2533.40 ms` with steps `2538.23`,
  `2531.50`, and `2535.30 ms`, about `10.1%` faster than the supplied
  llm.c baseline average (`2818.23 ms`). This is a fallback ceiling, not a
  pure-TK promotion: the focused pure-TK benchmark still shows the material
  dWeight and LM-head rows behind cuBLASLt, so the kernel-outperformance goal
  remains open.
- Rejected direct-Bcol dInput `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=2` under
  the O3 source-default stack. The macro build passed `test_attention` and
  `test_matmul` (`10/10`), but the focused benchmark showed only isolated
  wins for attention-projection dInput (`354.15 us` versus cuBLASLt
  `365.40 us`) and FC-projection forward (`1417.65 us` versus `1479.35 us`).
  The broad kernel target still failed: qkv dInput was `1018.97 us` versus
  `1007.21 us`, qkv dWeight was `1186.68 us` versus `1031.02 us`,
  attention-projection dWeight was `502.13 us` versus `327.35 us`,
  FC-projection dInput was `1559.36 us` versus `1365.63 us`, and all LM-head
  rows stayed slower than cuBLASLt. TinyStories 3-step validation averaged
  `2624.11 ms` with steps `2617.82`, `2621.74`, and `2626.48 ms`, so the
  source keeps direct-Bcol `SUPER_M=8`.
- Rejected direct-Bcol dInput `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=3` under
  the O3 source-default stack. The macro build passed `test_attention` and
  `test_matmul` (`10/10`), and TinyStories 3-step validation averaged
  `2619.77 ms` with steps `2616.35`, `2617.85`, and `2621.69 ms`. The focused
  benchmark had isolated wins for qkv dInput (`1018.38 us` versus cuBLASLt
  `1052.81 us`) and FC-projection forward (`1418.24 us` versus `1482.08 us`),
  but most targeted rows still failed the CUDA comparison: qkv dWeight was
  `1190.61 us` versus `986.20 us`, attention-projection dWeight was
  `469.43 us` versus `329.41 us`, FC-projection dInput was `1476.88 us`
  versus `1364.70 us`, and LM-head forward/dInput/dWeight remained slower than
  cuBLASLt. The trainer improvement was not promoted because the kernel goal
  requires every SM120 row to beat the CUDA variant, so the source keeps
  direct-Bcol `SUPER_M=8`.
- Rejected direct-Bcol dInput `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=4` under
  the O3 source-default stack. The macro build passed `test_attention` and
  `test_matmul` (`10/10`), but the focused benchmark worsened the direct-Bcol
  dInput rows that the swizzle targets: attention-projection dInput moved to
  `377.19 us` versus cuBLASLt `364.79 us`, and LM-head dInput moved to
  `22333.33 us` versus cuBLASLt `21136.92 us`. TinyStories 3-step validation
  averaged `2621.76 ms` with steps `2616.50`, `2618.77`, and `2624.75 ms`,
  another small end-to-end timing tie that still fails the per-kernel CUDA
  comparison, so the source keeps direct-Bcol `SUPER_M=8`.
- Rejected direct-Bcol dInput `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=5` under
  the O3 source-default stack. The macro build passed `test_attention` and
  `test_matmul` (`10/10`), and TinyStories 3-step validation improved to
  `2621.55 ms` with steps `2618.84`, `2618.99`, and `2624.12 ms`. The focused
  benchmark still failed the kernel goal: qkv dInput worsened to `1044.13 us`
  versus cuBLASLt `1007.17 us`, qkv dWeight stayed `1.21x` slower, LM-head
  forward stayed `1.12x` slower, and LM-head dInput stayed `1.05x` slower.
  The small trainer win was treated as noise/tradeoff rather than a promotion
  because the candidate moved key kernels farther from the CUDA variants.
- Rejected a qkv-only no-split dWeight route
  (`LLMK_SM120_QKV_DWEIGHT_SPLIT_K=1`). The temporary hook forced the
  `M == 3 * N` qkv dWeight shape off the default 8-way split-K fanout and
  added a guarded qkv-shaped dWeight smoke row; the build passed
  `test_attention` and `test_matmul` (`11/11`). The focused benchmark showed
  why split-K is required: qkv dWeight regressed to `2172.99 us` versus
  cuBLASLt `988.63 us`, with accumulated qkv dWeight at `2192.90 us` versus
  `990.11 us`. TinyStories 3-step validation averaged `2697.02 ms` with steps
  `2694.62`, `2693.42`, and `2700.62 ms`, much slower than the `2623.57 ms`
  O3 source default, so the temporary hook and smoke row were removed.
- Rejected a temporary 256x64 FC-projection dInput direct B-column route
  (`LLMK_SM120_DINP_FCPROJ_WIDE_DIRECT_BCOL=1`). The macro build passed
  `test_attention` and `test_matmul` (`11/11`, including the guarded FCProj
  wide direct-Bcol row), but the focused benchmark worsened FCProj dInput to
  `1654.04 us` versus cuBLASLt `1388.00 us` and left all material dWeight and
  LM-head rows behind cuBLASLt. TinyStories 3-step validation averaged
  `2623.74 ms` with steps `2618.41`, `2622.08`, and `2625.39 ms`, essentially
  tied with the `2623.57 ms` O3 source default but not a kernel win. The
  temporary alias, dispatch hook, and smoke row were removed.
- Profiled the selective SM120 cuBLASLt dInput+dWeight fallback with
  `LLMK_SM120_CUBLASLT_DINP_FALLBACK=1`,
  `LLMK_SM120_CUBLASLT_DWEIGHT_FALLBACK=1`, and
  `LLMK_SM120_PROFILE_TRAIN_STEP=1`. The 3-step TinyStories run averaged
  `2640.63 ms` with steps `2815.33`, `2639.21`, and `2642.04 ms`; the first
  profiled step paid extra fallback setup cost. Step-3 phase timings were:
  forward `805.23 ms`, FC-projection backward `326.01 ms`, FC backward
  `307.98 ms`, attention backward `283.61 ms`, qkv backward `231.32 ms`,
  LM-head backward `225.17 ms`, final-LayerNorm backward `146.55 ms`,
  classifier backward `86.23 ms`, and attention-projection backward
  `80.85 ms`. Compared with the pure-TK profile, the fallback mainly reduced
  FC-projection and attention-projection backward, while the forward bucket
  stayed dominant.
- Rejected an FC-only SM120 M3072/N128 dWeight swizzle route
  (`LLMK_SM120_DWEIGHT_M3072_SUPER_M=4`). The source candidate passed
  `test_attention`; the first `test_matmul` run hit the known transient
  MLP-up forward row failure, and an immediate rerun passed `10/10`. The
  focused benchmark worsened FC dWeight to `1500.39 us` versus cuBLASLt
  `1307.97 us` and left FC dWeight accumulate at `1456.16 us` versus
  `1331.24 us`. TinyStories 3-step validation averaged `2621.38 ms` with
  steps `2616.59`, `2618.81`, and `2623.94 ms`, but the small trainer win was
  treated as noise because the targeted kernel moved farther from the CUDA
  variant. The FC dWeight route remains on the shared N128 TN tile.
- Refreshed the current SM120 pure-TK phase profile with
  `LLMK_SM120_PROFILE_TRAIN_STEP=1`. The profiling build passed compilation and
  TinyStories 3-step validation averaged `2668.62 ms` with steps `2666.77`,
  `2665.92`, and `2671.33 ms`; the profiler overhead makes this unsuitable as
  a source-default timing comparison. Step-3 phase timings were: forward
  `807.13 ms`, FC-projection backward `339.81 ms`, FC backward `313.23 ms`,
  attention backward `287.65 ms`, qkv backward `234.50 ms`, LM-head backward
  `202.43 ms`, final-LayerNorm backward `166.98 ms`, classifier backward
  `88.86 ms`, and attention-projection backward `86.90 ms`. The next
  optimization focus remains the GEMM-heavy forward and FC backward buckets.
- Rejected a temporary SM120 huge-N `256x96x32` forward tile selector
  (`LLMK_SM120_HUGE_N_N96=1`) at the matmul smoke gate. The candidate compiled
  and `test_attention` passed all three smoke shapes, but `test_matmul` hit an
  illegal memory access on the GPT-2 LM-head forward row before benchmarking.
  No TinyStories validation was run; huge-N forward remains on the stable
  `256x128x32` tile.
- Rejected a temporary SM120 huge-N `256x384x32` forward tile selector
  (`LLMK_SM120_HUGE_N_N384=1`) at the matmul smoke gate. The candidate
  compiled and `test_attention` passed all three smoke shapes, but
  `test_matmul` aborted on the GPT-2 LM-head row while creating the tensor map
  for the `384x32` B tile: ThunderKittens asserted `smem_shape[1] <= 256`.
  No benchmark or TinyStories validation was run; huge-N forward remains on
  the valid `256x128x32` tile.
- Rejected a temporary SM120 huge-N `256x256x32` forward tile selector
  (`LLMK_SM120_HUGE_N_N256=1`) at the matmul smoke gate. The candidate
  compiled and `test_attention` passed all three smoke shapes, but
  `test_matmul` aborted on the GPT-2 LM-head row because padded vocab
  `N=50304` is not divisible by the candidate `N_TILE=256`. No benchmark or
  TinyStories validation was run; huge-N forward remains on the valid
  `256x128x32` tile.
- Rejected SM120 `LLMK_SM120_HUGE_N_K_TILE=24` at compile time for the huge-N
  LM-head forward tile. ThunderKittens BF16 shared/register tile constraints
  require rows/columns to be divisible by the base tile dimensions, and the
  K24 build failed static assertions such as `Cols must be divisible by the
  tile dimension`, `Rows must be divisible by the tile dimension`, and
  `Columns must be divisible by the tile size` while instantiating the
  `matmul_huge_n_*` aliases. No smoke or TinyStories validation was possible,
  so huge-N forward stays on `LLMK_SM120_HUGE_N_K_TILE=32`.
- Rejected a qkv-only SM120 M2304/N128 dWeight swizzle route
  (`LLMK_SM120_DWEIGHT_M2304_SUPER_M=3`). The source candidate passed
  `test_attention` and `test_matmul`, but the focused benchmark worsened qkv
  dWeight to `1231.58 us` versus cuBLASLt `987.56 us` and qkv dWeight
  accumulate to `1195.82 us` versus `997.37 us`. TinyStories 3-step validation
  averaged `2624.33 ms` with steps `2617.99`, `2620.54`, and `2628.12 ms`,
  slower than the `2623.57 ms` O3 source default, so the qkv dWeight route
  remains on the shared N128 TN tile.
- Rejected a shape-specific SM120 M768/N128 dWeight swizzle route
  (`LLMK_SM120_DWEIGHT_M768_SUPER_M=3`). The source candidate passed
  `test_attention` and `test_matmul`, and TinyStories 3-step validation
  averaged `2621.36 ms` with steps `2615.21`, `2618.79`, and `2623.94 ms`.
  Focused kernel timings did not justify the route: attproj dWeight was
  `466.96 us` versus cuBLASLt `347.29 us`, attproj dWeight accumulate was
  `467.72 us` versus `330.04 us`, fcproj dWeight was `1452.33 us` versus
  `1352.03 us`, and fcproj dWeight accumulate was `1457.80 us` versus
  `1354.56 us`. The small trainer improvement was treated as noise because
  the targeted kernels remained slower than both the source default and the
  CUDA variants, so the route was removed.
- Rejected SM120 `LLMK_SM120_DINP_SUPER_M=7` for the non-direct dInput and
  dGELU routes. The macro build passed `test_attention`, but `test_matmul`
  failed the default dInput A*B row with max abs diff `6.5547` against the
  `0.50` tolerance, so no TinyStories validation was run and the dInput
  swizzle stays at `8`.
- Rejected an O3 retest of SM120 `LLMK_SM120_HUGE_N_K_TILE=16` for the
  huge-N LM-head routes. The macro build passed `test_attention` and
  `test_matmul`, then TinyStories 3-step validation averaged `2633.06 ms`
  with steps `2627.45`, `2630.26`, and `2635.86 ms`, slower than the
  `2623.57 ms` O3 source default, so the huge-N K tile stays at `32`.
- Rejected routing the SM120 huge-N dWeight TN alias through
  `LLMK_SM120_DWEIGHT_SUPER_M` instead of the global `LLMK_SM120_SUPER_M`.
  The source candidate passed `test_attention` and `test_matmul`, then
  TinyStories 3-step validation averaged `2623.71 ms` with steps `2616.50`,
  `2620.19`, and `2627.22 ms`, a near tie but still slower than the
  `2623.57 ms` O3 source default, so the huge-N TN swizzle stays unchanged.
- Rejected SM120 `LLMK_SM120_DWEIGHT_N128_K_TILE=8` at compile time. The
  N128 dWeight K-tile shrink trips ThunderKittens shared/register tile static
  assertions because 8-row swizzled BF16 tiles are not divisible by the base
  tile row dimension, so no smoke or TinyStories validation was possible.
- Rejected an O3 retest of SM120 `LLMK_SM120_DWEIGHT_SUPER_M=1` for the
  TN dWeight kernels. The macro build passed `test_attention` and
  `test_matmul`, then TinyStories 3-step validation averaged `2625.48 ms`
  with steps `2618.77`, `2621.77`, and `2629.18 ms`, slower than the
  `2623.57 ms` O3 source default, so the dWeight swizzle stays at `2`.
- Rejected an O3 retest of SM120 `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=7`
  for the direct B-column dInput route. The macro build passed
  `test_attention` and `test_matmul`, then TinyStories 3-step validation
  averaged `2626.42 ms` with steps `2620.85`, `2622.70`, and `2630.14 ms`,
  slower than the `2623.57 ms` O3 source default, so the dInput direct B-column
  swizzle stays at `8`.
- Rejected an O3 retest of global SM120 `LLMK_SM120_SUPER_M=8` on top of
  the current pure-TK source default. The macro build passed `test_attention`
  and `test_matmul`, then TinyStories 3-step validation averaged `2629.76 ms`
  with steps `2619.08`, `2625.43`, and `2634.09 ms`, slower than the
  `2623.57 ms` O3 source default, so `LLMK_SM120_SUPER_M` stays at `7`.
- Rejected a temporary streaming cache-policy hook for the SM120 split-K
  dWeight partial reducer. The macro build
  (`LLMK_SM120_REDUCE_PARTIALS_STREAMING=1`) used streaming 128-bit loads for
  partials and `store128cg` for the final write, passed `test_attention`, and
  passed `test_matmul` on rerun after a transient MLP-up forward row. TinyStories
  3-step validation averaged `2628.73 ms` with steps `2620.31`, `2626.60`, and
  `2630.86 ms`, slower than the `2623.57 ms` O3 source default, so the hook was
  removed.
- Rejected raising the SM120 huge-N threshold to force LM-head forward off the
  huge-N tile. The pure-TK macro build
  (`LLMK_SM120_HUGE_N_THRESHOLD=65536`) passed `test_attention`, but
  `test_matmul` failed LM-head forward on both runs and also hit MLP-up forward
  on the first run. No TinyStories validation was run for the invalid output
  candidate; the huge-N threshold remains `8192`.
- Rejected disabling the SM120 N128 dWeight TN tile route under the O3
  source-default stack. The pure-TK macro build (`LLMK_SM120_DWEIGHT_N128=0`)
  passed `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes),
  but TinyStories 3-step validation averaged `2694.55 ms` with steps
  `2690.18`, `2690.88`, and `2698.23 ms`, much slower than the `2623.57 ms`
  O3 source default. The source keeps `LLMK_SM120_DWEIGHT_N128=1`.
- Rejected disabling the SM120 forward N96 route under the O3 source-default
  stack. The pure-TK macro build (`LLMK_SM120_FORWARD_N96=0`) passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes), but
  TinyStories 3-step validation averaged `2679.16 ms` with steps `2672.93`,
  `2677.94`, and `2680.38 ms`, slower than the `2623.57 ms` O3 source
  default. The source keeps `LLMK_SM120_FORWARD_N96=1`.
- Rejected retesting the SM120 huge-N K64 tile under the O3 source-default
  stack. The pure-TK macro build (`LLMK_SM120_HUGE_N_K_TILE=64`) passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes), but
  TinyStories 3-step validation averaged `2627.45 ms` with steps `2617.26`,
  `2625.20`, and `2629.70 ms`, slower than the `2623.57 ms` O3 source
  default. Huge-N forward keeps `LLMK_SM120_HUGE_N_K_TILE=32`.
- Rejected serializing SM120 split-K dWeight partial launches onto the main
  stream. The pure-TK macro build (`LLMK_SM120_DWEIGHT_SPLIT_K_STREAMS=0`)
  passed `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes),
  but TinyStories 3-step validation averaged `3021.54 ms` with steps
  `3012.40`, `3020.36`, and `3022.72 ms`, far slower than the `2623.57 ms`
  O3 source default. Split-K dWeight keeps its nonblocking side-stream fanout.
- Rejected retesting pure SM120 TK codegen with `-Xptxas=-dlcm=cg` under the
  O3 source-default stack. The build warned that the explicit `cg` value
  overrode the default `-dlcm=ca`, passed `test_attention`, and passed
  `test_matmul` on rerun after an LM-head forward smoke failure. TinyStories
  3-step validation averaged `3285.68 ms` with steps `3280.19`, `3281.14`,
  and `3290.23 ms`, far slower than the `2623.57 ms` O3 source default. The
  SM120 pure-TK codegen default remains `-Xptxas=-dlcm=ca`.
- Rejected disabling SM120 dWeight direct accumulation after the TN direct
  B-column promotion. The pure-TK macro build
  (`LLMK_SM120_DWEIGHT_DIRECT_ACCUM=0`) passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  averaged `2626.56 ms` with steps `2615.71`, `2622.95`, and `2630.16 ms`,
  slightly slower than the `2623.57 ms` O3 source default. Accumulated
  dWeight stays on the direct in-kernel accumulation path.
- Rejected enabling the SM120 backward N96 route for 768-wide dInput/dWeight
  rows. The pure-TK macro build (`LLMK_SM120_BACKWARD_N96=1`) passed
  `test_attention`; the first `test_matmul` hit the known transient MLP-up row
  and the immediate rerun passed `10/10`. TinyStories 3-step validation
  averaged `2721.11 ms` with steps `2711.96`, `2719.10`, and `2723.12 ms`,
  much slower than the `2623.57 ms` O3 source default, so
  `LLMK_SM120_BACKWARD_N96` remains disabled.
- Refreshed the focused SM120 matmul benchmark for the current pure-TK `-O3`
  source default. The benchmark still shows the remaining cuBLASLt gaps:
  qkv dW `1233.01 us` versus `987.85 us`, attention-projection dW
  `470.28 us` versus `327.40 us`, FC dW `1451.49 us` versus `1351.60 us`,
  FC-projection dW `1505.57 us` versus `1333.43 us`, LM-head forward
  `24737.11 us` versus `22250.03 us`, and LM-head dInput `22408.23 us`
  versus `21037.82 us`. These rows remain the next optimization targets.
- Rebaselined the current pure SM120 TK source default with the Makefile's
  actual `-O3` setting (`DEVICE_ARCH=SM120`, `SM120_USE_CUBLASLT_GEMM=0`,
  no `FORCE_NVCC_O=0`). The build passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes). TinyStories 3-step validation
  averaged `2623.57 ms` with steps `2615.57`, `2621.21`, and `2625.93 ms`,
  matching the prior `2623.34 ms` default within normal short-run noise.
- Rejected disabling the SM120 huge-N 256x128 forward tile after the TN direct
  B-column promotion. The pure-TK macro build
  (`LLMK_SM120_HUGE_N_M256=0`, `SM120_USE_CUBLASLT_GEMM=0`) passed
  `test_attention`, but `test_matmul` failed the LM-head forward row on the
  first pass and failed both MLP-up and LM-head forward on the immediate rerun.
  No TinyStories validation was run for the invalid output candidate; huge-N
  forward stays on `LLMK_SM120_HUGE_N_M256=1`.
- Rejected lowering dWeight split-K to 4 on top of the promoted TN direct
  B-column source. The pure-TK macro build
  (`LLMK_SM120_DWEIGHT_SPLIT_K=4`, `SM120_USE_CUBLASLT_GEMM=0`) passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes), but
  TinyStories 3-step validation averaged `2637.24 ms` with steps `2635.10`,
  `2633.56`, and `2640.91 ms`, slower than the `2623.34 ms` source default.
  The source keeps the accepted dWeight split-K default.
- Rejected retesting qkv dWeight split-K 16 on top of the promoted TN direct
  B-column source. The pure-TK macro build
  (`LLMK_SM120_DWEIGHT_SPLIT_K=16`, `SM120_USE_CUBLASLT_GEMM=0`) passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes), but
  TinyStories 3-step validation averaged `2628.21 ms` with steps `2621.22`,
  `2626.84`, and `2629.59 ms`, slower than the `2623.34 ms` source default.
  dWeight split-K remains at the accepted source default.
- Rejected retesting SM120 dWeight N128 K-tile 32 on top of the promoted TN
  direct B-column source. The pure-TK macro build
  (`LLMK_SM120_DWEIGHT_N128_K_TILE=32`, `SM120_USE_CUBLASLT_GEMM=0`) passed
  `test_attention`; the first `test_matmul` run hit the known transient MLP-up
  row and the immediate rerun passed `10/10`. TinyStories 3-step validation
  averaged `2645.25 ms` with steps `2635.58`, `2643.76`, and `2646.74 ms`,
  slower than the `2623.34 ms` source default, so N128 dWeight remains on
  `LLMK_SM120_DWEIGHT_N128_K_TILE=16`.
- Rejected a temporary SM120 TN direct A-column register-load hook for
  dWeight. The pure-TK macro build
  (`LLMK_SM120_TN_DIRECT_A_COL=1`, `SM120_USE_CUBLASLT_GEMM=0`) passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes), but
  TinyStories 3-step validation regressed to `2689.56 ms` with steps
  `2679.82`, `2684.98`, and `2694.15 ms`, slower than the `2623.34 ms`
  source default. The focused benchmark also left material dWeight rows behind
  cuBLASLt (for example qkv dW `1376.42 us` versus `987.99 us` and
  attention-projection dW `528.20 us` versus `329.12 us`), so the hook was
  removed. An earlier `DEVICE_ARCH=SM120` trainer run averaged `2507.56 ms`
  but was excluded because the Makefile default enabled cuBLASLt GEMM dispatch.
- Rejected retesting `LLMK_SM120_DWEIGHT_SUPER_M=4` on top of the promoted TN
  direct B-column dWeight source. The macro build passed `test_matmul`
  (`10/10`) and `test_attention` (all three smoke shapes), but TinyStories
  3-step validation averaged `2625.32 ms` with steps `2617.10`, `2622.08`,
  and `2628.55 ms`, slower than the `2623.34 ms` source default. The dWeight
  swizzle remains `LLMK_SM120_DWEIGHT_SUPER_M=2`.
- Rejected retesting `LLMK_SM120_DWEIGHT_SUPER_M=3` on top of the promoted TN
  direct B-column dWeight source. The macro build passed `test_matmul`
  (`10/10`) and `test_attention` (all three smoke shapes), but TinyStories
  3-step validation averaged `2624.35 ms` with steps `2616.62`, `2620.31`,
  and `2628.38 ms`, slightly slower than the new `2623.34 ms` source default.
  The dWeight swizzle remains `LLMK_SM120_DWEIGHT_SUPER_M=2`.
- Promoted SM120 TN dWeight direct B-column register loads
  (`LLMK_SM120_TN_DIRECT_B_COL=1`). The macro probe skipped the B-side
  register-layout swap inside the `A^T*B` kernel, passed `test_matmul`
  (`10/10`) and `test_attention` (all three smoke shapes), and improved
  TinyStories 3-step validation to `2622.90 ms` with steps `2615.88`,
  `2618.16`, and `2627.64 ms`. After promoting the macro as the source
  default, the no-override rebuild passed `test_attention`; the first
  `test_matmul` hit the known transient MLP-up row and the immediate rerun
  passed `10/10`. No-override TinyStories validation averaged `2623.34 ms`
  with steps `2616.50`, `2619.28`, and `2627.41 ms`, substantially improving
  the previous `2653.36 ms` pure-TK source default. The focused benchmark
  showed the intended dWeight gains, including qkv dW `1211.48 us` versus the
  prior `1250.10 us`, FC dW `1451.95 us` versus `1542.57 us`, FC-projection
  dW `1463.00 us` versus `1509.03 us`, and LM-head dW `22053.86 us` versus
  `23072.15 us`; dWeight rows still trail cuBLASLt, so the SM120 optimization
  goal remains open.
- Rejected pure SM120 TK codegen with
  `-Xptxas=--allow-expensive-optimizations=true`. The build passed
  `test_attention` and eventually passed `test_matmul` (`10/10`) after two
  transient smoke failures, but TinyStories 3-step validation averaged
  `2655.86 ms` with steps `2652.09`, `2654.48`, and `2657.24 ms`, slower than
  the `2653.36 ms` source default. The SM120 pure-TK compiler defaults remain
  `-Xptxas=-dlcm=ca --extra-device-vectorization`.
- Rejected a temporary fixed-8 split-K dWeight partial reducer specialization.
  The source hook replaced the runtime `parts` loop with an unrolled
  `PARTS=8` CUDA kernel for the common qkv/FC/FC-projection dWeight case. The
  first `test_matmul` pass failed the accumulated dWeight row, the immediate
  rerun passed `10/10`, and `test_attention` passed all three smoke shapes.
  TinyStories 3-step validation averaged `2655.62 ms` with steps `2649.45`,
  `2652.74`, and `2658.50 ms`, slower than the `2653.36 ms` source default,
  so the temporary reducer specialization was removed.
- Rejected a post-vectorization LM-head dWeight 2-way split-K retest with
  `LLMK_SM120_DEFER_LMHEAD_DWEIGHT=0` and
  `LLMK_SM120_LARGE_DWEIGHT_SPLIT_K=2`. The macro build passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes), but
  it raised activation allocation from `25514 MiB` to `25588 MiB` and
  TinyStories 3-step validation averaged `2656.63 ms` with steps `2651.01`,
  `2653.46`, and `2659.81 ms`, slower than the `2653.36 ms` source default.
  LM-head dWeight stays on the deferred one-part side-stream route.
- Rejected lowering the direct B-column dInput cap back to
  `LLMK_SM120_DINP_DIRECT_BCOL_K_CAP=2304` after the SM120 compiler default
  promotions. The macro build passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  regressed to `2665.76 ms` with steps `2661.76`, `2663.80`, and `2667.72 ms`
  and shifted the early norm trace. The source keeps the accepted `3072` cap so
  FC-projection dInput remains on the direct B-column route.
- Rejected a temporary large-K-only direct B-column dInput swizzle split with
  `LLMK_SM120_DINP_DIRECT_BCOL_LARGEK_SUPER_M=6`. The source hook added a
  separate LM-head-style large-K direct-B-column alias while leaving small-K
  dInput on the accepted `SUPER_M=8` route. The macro build passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes), but
  TinyStories 3-step validation averaged `2654.99 ms` with steps `2648.38`,
  `2653.73`, and `2656.25 ms`, slower than the `2653.36 ms` source default.
  The temporary split hook was removed.
- Rejected disabling the promoted large-K direct B-column dInput route with
  `LLMK_SM120_DINP_DIRECT_BCOL_LARGEK=0` after the SM120 compiler default
  promotions. The macro build passed `test_matmul` (`9/9`, with the guarded
  large-K row compiled out) and `test_attention` (all three smoke shapes), but
  TinyStories 3-step validation regressed to `2670.30 ms` with steps
  `2662.88`, `2666.78`, and `2673.82 ms` and shifted the early norm trace.
  Large-K direct B-column dInput remains enabled.
- Rejected `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=11` after the SM120 compiler
  default promotions. The macro build passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  averaged `2656.41 ms` with steps `2650.07`, `2654.88`, and `2657.93 ms`,
  slower than the `2653.36 ms` source default. The direct B-column swizzle
  neighborhood keeps the accepted `SUPER_M=8` source default.
- Rejected `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=10` after the SM120 compiler
  default promotions. The first `test_matmul` pass hit the known transient
  GPT-2 MLP-up row, the immediate rerun passed `10/10`, and `test_attention`
  passed all three smoke shapes. TinyStories 3-step validation averaged
  `2655.51 ms` with steps `2650.55`, `2654.57`, and `2656.45 ms`, slower than
  the `2653.36 ms` source default, so direct B-column dInput remains at
  `SUPER_M=8`.
- Rejected `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=9` after the SM120 compiler
  default promotions. The macro build passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  averaged `2657.09 ms` with steps `2650.84`, `2654.55`, and `2659.62 ms`,
  slower than the `2653.36 ms` source default. Direct B-column dInput keeps the
  `SUPER_M=8` swizzle.
- Rejected retesting `LLMK_SM120_ATTN_BWD_BLOCK=32` after the SM120
  cache-policy and extra-device-vectorization promotions. The macro build
  passed `test_matmul` (`10/10`) and `test_attention` (all three smoke
  shapes), but TinyStories 3-step validation regressed to `2664.34 ms` with
  steps `2659.36`, `2661.76`, and `2666.92 ms`, slower than the `2653.36 ms`
  source default. Packed-QKV attention backward remains on the 16-row tile.
- Rejected retesting `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=7` after the
  cache-policy and extra-device-vectorization promotions. The macro build
  passed `test_matmul` (`10/10`) and `test_attention` (all three smoke
  shapes), but TinyStories 3-step validation averaged `2654.74 ms` with
  steps `2649.12`, `2651.13`, and `2658.35 ms`, slower than the `2653.36 ms`
  source default. The direct B-column dInput route keeps
  `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=8`.
- Rejected retesting pure SM120 TK codegen with `FORCE_NVCC_O=2` after the
  cache-policy and extra-device-vectorization promotions. The O2 build passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes), but
  TinyStories 3-step validation regressed to `2656.56 ms` with steps
  `2648.49`, `2655.14`, and `2657.98 ms`, slower than the `2653.36 ms` O3
  source default. Pure SM120 TK builds keep `FORCE_NVCC_O=3`.
- Rejected retesting `LLMK_SM120_DWEIGHT_SPLIT_K=16` on top of the promoted
  SM120 pure-TK compiler defaults. The build passed `test_matmul` (`10/10`)
  and `test_attention` (all three smoke shapes), but TinyStories 3-step
  validation averaged `2656.41 ms` with steps `2649.44`, `2653.55`, and
  `2659.27 ms`, slower than the new no-override source default
  (`2653.36 ms`). dWeight split-K remains `8`.
- Refreshed the focused SM120 matmul benchmark after promoting the pure-TK
  compiler defaults. The run confirmed the new defaults improve overall step
  time but do not close the remaining GEMM gaps: qkv dWeight stayed
  `1250.10 us` TK versus `987.96 us` cuBLASLt, attention-projection dWeight
  `475.81 us` versus `327.49 us`, FC dWeight `1542.57 us` versus
  `1308.14 us`, FC-projection dWeight `1509.03 us` versus `1352.25 us`, and
  LM-head forward/dInput/dWeight stayed around `1.07x` to `1.13x` slower than
  cuBLASLt. Backward GEMM and LM-head scheduling remain the active targets.
- Promoted `--extra-device-vectorization` alongside the SM120 pure-TK
  `-Xptxas=-dlcm=ca` default. The explicit override build passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes), then
  averaged `2652.17 ms` over 3 TinyStories steps with steps `2647.16`,
  `2651.45`, and `2652.88 ms`. After promoting the flag into the Makefile, the
  no-override rebuild passed the same smokes and validated at `2653.36 ms`
  with steps `2648.73`, `2649.81`, and `2656.92 ms`, improving the
  `2654.56 ms` cache-policy-only default while still trailing the backward
  cuBLASLt fallback target.
- Rejected overriding the promoted SM120 pure-TK cache policy with
  `-Xptxas=-dlcm=cg`. The override build emitted the expected ptxas warning
  that the later `cg` value won over the default `ca`, passed `test_matmul`
  (`10/10`) and `test_attention` (all three smoke shapes), but TinyStories
  3-step validation regressed sharply to `3314.65 ms` with steps `3311.07`,
  `3310.12`, and `3319.19 ms`. The source default remains
  `-Xptxas=-dlcm=ca` for SM120 pure-TK builds.
- Promoted `-Xptxas=-dlcm=ca` for SM120 pure-TK builds. The explicit compiler
  cache-policy probe passed `test_matmul` (`10/10`) and `test_attention` (all
  three smoke shapes), then averaged `2654.23 ms` over 3 TinyStories steps
  with steps `2649.17`, `2651.59`, and `2656.86 ms`. After promoting the flag
  into the Makefile's SM120 pure-TK defaults, the no-override rebuild passed
  `test_attention`; the first `test_matmul` hit the intermittent MLP-up row but
  an immediate rerun passed `10/10`. No-override TinyStories validation
  averaged `2654.56 ms` with steps `2649.19`, `2654.38`, and `2654.74 ms`,
  a small improvement over the previous `2654.93 ms` source default while the
  backward cuBLASLt fallback target remains open.
- Rejected the narrower SM120 TK LayerNorm forward-only probe that kept the
  CUDA backward helper on `warpReduceSum` instead of the TK warp reduction.
  The build again passed `test_matmul` (`10/10`) and `test_attention` (all
  three smoke shapes), and the LayerNorm forward/fused-forward values stayed
  within tolerance. `test_layernorm` still failed the backward `dbias` row at
  the same `0.496948` max diff versus the `0.120` tolerance, so the temporary
  SM120 opt-in gate was removed and no TinyStories validation was run.
- Rejected enabling the existing TK LayerNorm forward wrapper for SM120 behind
  an opt-in `LLMK_SM120_TK_LAYERNORM=1` gate. The probe built cleanly and the
  LayerNorm smoke showed the SM120 TK forward outputs were within tolerance
  (`0.020295` max diff for plain forward and `0.024888` for fused residual
  output), while `test_matmul` (`10/10`) and `test_attention` (all three smoke
  shapes) still passed. `test_layernorm` failed the backward `dbias` check,
  however, with `0.496948` max diff versus the `0.120` tolerance, so no
  TinyStories validation was run and the temporary SM120 gate was removed.
- Rejected a temporary M=768 dWeight 128x64 K16 route at the smoke gate. The
  guarded `LLMK_SM120_DWEIGHT_M768_N64_K16=1` probe sent M=768, N%64 TN
  dWeight rows through a 128x64 tile to target attention-projection and
  FC-projection dWeight. The focused benchmark improved some dWeight rows
  versus the promoted default, including FC-projection dWeight (`1464.28 us`
  TK versus `1373.98 us` cuBLASLt) and qkv dWeight (`1240.12 us` versus
  `987.21 us`), but worsened accumulated attention-projection dWeight
  (`521.40 us` versus `329.94 us`) and remained behind cuBLASLt overall.
  `test_attention` passed all three smoke shapes, but `test_matmul` failed the
  GPT-2 MLP-up forward row (`6.4688` max diff versus `0.50` tolerance), so no
  TinyStories validation was run and the temporary route was removed.
- Rejected a temporary non-qkv dWeight split-K cap-16 hook at the smoke gate.
  The candidate added a guarded `LLMK_SM120_NON_QKV_DWEIGHT_SPLIT_K_CAP=16`
  path and resized trainer/benchmark scratch so attention-projection, FC, and
  FC-projection dWeight rows could actually launch 16 split-K parts while qkv
  stayed at the source split. The focused benchmark was mixed and worsened the
  targeted attention-projection dWeight row (`497.45 us` TK versus
  `327.57 us` cuBLASLt) plus accumulated dWeight (`538.83 us` versus
  `330.99 us`). `test_attention` passed all three smoke shapes, but
  `test_matmul` failed the MLP-up forward row (`7.3281` max diff) and the
  accumulated dWeight row (`2.6250` max diff), so no TinyStories validation was
  run and the temporary hook was removed.
- Promoted `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=8` after retesting the
  direct B-column dInput swizzle on top of the accepted `K_CAP=3072` source.
  The macro retest passed `test_matmul` (`10/10`) and `test_attention` (all
  three smoke shapes), then averaged `2653.53 ms` over 3 TinyStories steps.
  After promoting the source default from `7` to `8`, the no-override build
  passed the same smokes and validated at `2654.93 ms` with steps `2648.42`,
  `2650.46`, and `2659.41 ms`, improving the previous source default
  (`2657.45 ms`) while leaving the backward cuBLASLt fallback target open.
- Rejected retesting `LLMK_SM120_DWEIGHT_SPLIT_K=16` on top of the promoted
  FC-projection dInput direct route. The macro build passed `test_matmul`
  (`10/10`) and `test_attention` (all three smoke shapes), but TinyStories
  3-step validation regressed to `2660.48 ms` with steps `2652.14`, `2659.02`,
  and `2661.93 ms`, slower than the new no-override source default
  (`2657.45 ms`). dWeight split-K remains `8`.
- Refreshed SM120 per-step profiling after promoting the FC-projection dInput
  direct B-column route. The `LLMK_SM120_PROFILE_TRAIN_STEP=1` build completed
  the required TinyStories 3-step validation at `2703.41 ms` average with
  steps `2699.10`, `2703.02`, and `2703.81 ms`; the profiling events explain
  the overhead versus the unprofiled `2657.45 ms` source default. The dominant
  steady-state buckets are forward (`~805 ms`), FC projection backward
  (`~346 ms`), FC backward (`~320 ms`), attention backward (`~284 ms`), QKV
  backward (`~241 ms`), LM-head backward (`~205 ms`), and final LayerNorm
  backward (`~178 ms`). Backward GEMM scheduling remains the next optimization
  target.
- Promoted the SM120 direct B-column dInput route to cover the GPT-2
  FC-projection dInput shape and use a dedicated `SUPER_M=7` swizzle. The
  macro probe combined `LLMK_SM120_DINP_DIRECT_BCOL_K_CAP=3072` with
  `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=7`, passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes), and averaged `2656.69 ms` over 3
  TinyStories steps. After promoting the defaults in source, the no-override
  build passed the same smokes and validated at `2657.45 ms` with steps
  `2650.13`, `2654.87`, and `2660.02 ms`, improving the pure-TK source
  baseline from `2663.76 ms` while still leaving the combined backward
  cuBLASLt fallback target at `2589.00 ms`.
- Refreshed the combined current-stack SM120 dInput+dWeight cuBLASLt fallback
  diagnostic. The build used both `LLMK_SM120_CUBLASLT_DINP_FALLBACK=1` and
  `LLMK_SM120_CUBLASLT_DWEIGHT_FALLBACK=1`, leaving only forward GEMM on the
  promoted pure-TK stack. It passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes), then TinyStories 3-step
  validation averaged `2589.00 ms` with steps `2604.40`, `2587.43`, and
  `2590.58 ms`. This is faster than the cached full cuBLASLt fallback near
  `2623 ms`, confirming the remaining pure-TK gap is dominated by backward
  dInput/dWeight GEMMs rather than the current K32 forward path.
- Rejected a direct B-column fused-dGELU dInput route for the GPT-2 FC
  backward shape after pairing it with exact dGELU tanh to recover the
  previous smoke failure. The temporary
  `LLMK_SM120_DINP_DIRECT_BCOL_DGELU_FC=1` hook added a guarded FC dInput smoke
  row and was built with `LLMK_SM120_APPROX_DGELU_TANH=0`; that build passed
  `test_matmul` (`11/11`) and `test_attention` (all three smoke shapes).
  TinyStories 3-step validation still regressed to `2665.14 ms` with steps
  `2658.41`, `2663.26`, and `2667.02 ms`, so the temporary alias, dispatch
  hook, and smoke row were removed.
- Refreshed the current-stack SM120 dInput-only cuBLASLt fallback diagnostic.
  The build used `LLMK_SM120_CUBLASLT_DINP_FALLBACK=1` with explicit cuBLASLt
  linkage while leaving forward and dWeight on the promoted pure-TK stack. It
  passed `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes),
  then TinyStories 3-step validation averaged `2617.42 ms` with steps
  `2611.05`, `2615.34`, and `2619.50 ms`. This beats the current pure-TK
  source baseline (`2663.76 ms`) and the cached full cuBLASLt fallback near
  `2623 ms`, making dInput the immediate SM120 TK optimization target ahead of
  further dWeight work.
- Refreshed the current-stack SM120 dWeight-only cuBLASLt fallback diagnostic.
  The build used `LLMK_SM120_CUBLASLT_DWEIGHT_FALLBACK=1` with explicit
  cuBLASLt linkage while leaving forward and dInput on the promoted pure-TK
  stack. It passed `test_matmul` (`10/10`) and `test_attention` (all three
  smoke shapes), then TinyStories 3-step validation averaged `2626.98 ms` with
  steps `2637.50`, `2625.68`, and `2628.28 ms`. This beats the current
  pure-TK source baseline (`2663.76 ms`) and lands close to the cached full
  cuBLASLt fallback near `2623 ms`, confirming dWeight as the highest-value
  SM120 kernel target after the K32 huge-N promotion.
- Rejected a current-stack forward-only cuBLASLt fallback refresh at the smoke
  gate. The diagnostic build used
  `LLMK_SM120_CUBLASLT_FORWARD_FALLBACK=1` with explicit cuBLASLt linkage on
  top of the promoted K32 huge-N source, and `test_attention` passed all three
  smoke shapes. `test_matmul` failed before training, however: the accumulated
  dWeight row reported max diff `3.1250` versus the `0.50` tolerance, leaving
  only `9/10` rows passing. No TinyStories 3-step validation was run for this
  invalid fallback binary; the pure-TK default remains the last valid
  `2663.76 ms` source baseline, still chasing the cached full cuBLASLt
  fallback near `2623 ms`.
- Rejected deferring the LM-head dWeight side-stream wait past the first
  FC-projection backward. The temporary
  `LLMK_SM120_DEFER_LMHEAD_DWEIGHT_PAST_FCPROJ=1` hook avoided reusing
  `acts.output` as the first FCProj dbias scratch buffer, then waited after
  that FCProj backward to overlap more of the LM-head TN dWeight work. The
  macro build passed `test_matmul` (`10/10`) and `test_attention` (all three
  smoke shapes), but TinyStories 3-step validation regressed to `2667.49 ms`
  total average with steps `2666.27`, `2665.81`, and `2669.16 ms`. The
  temporary scheduling hook was removed.
- Refreshed current-default SM120 per-step profiling after the decoupled K32
  huge-N promotion. The `LLMK_SM120_PROFILE_TRAIN_STEP=1` build completed the
  required TinyStories 3-step validation with steps `2709.38`, `2709.39`, and
  `2711.79 ms` (`2710.59 ms` total average); the extra profiling events
  account for the overhead versus the unprofiled `2663.76 ms` source default.
  The dominant steady-state buckets are forward (`~804 ms`), FC projection
  backward (`~346 ms`), FC backward (`~330 ms`), attention backward
  (`~284 ms`), QKV backward (`~241 ms`), LM-head backward (`~204 ms`), and
  final LayerNorm backward (`~178 ms`). The next optimization focus remains
  GEMM/attention-heavy work, not optimizer/update overhead.
- Rejected `LLMK_SM120_FORWARD_N96=0` under the decoupled K32 default stack.
  The macro build passed `test_matmul` (`10/10`) and `test_attention` (all
  three smoke shapes), but the focused benchmark confirmed that the 128x96
  forward tile is still needed: qkv forward regressed to `1241.34 us`,
  attention-projection forward to `435.84 us`, FC fused forward to
  `1796.82 us`, FC projection forward to `1626.94 us`, and LM-head forward to
  `25152.47 us`. TinyStories 3-step validation regressed to `2718.36 ms` total
  average with steps `2717.12`, `2716.66`, and `2720.06 ms`, so
  `LLMK_SM120_FORWARD_N96` stays enabled.
- Rejected non-huge forward `LLMK_SM120_K_TILE=64` on top of the decoupled
  K32 huge-N default. The macro build passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes), but the focused benchmark
  regressed the non-huge forward rows badly: qkv forward `1328.62 us`,
  attention-projection forward `465.15 us`, FC fused forward `1959.33 us`, and
  FC projection forward `1823.91 us`. TinyStories 3-step validation regressed
  to `2741.43 ms` total average with steps `2739.89`, `2738.96`, and
  `2743.90 ms`, so the non-huge forward K tile stays at `32`.
- Rejected `LLMK_SM120_HUGE_N_K_TILE=64` after the decoupled K32 promotion.
  The macro build kept N128 dWeight isolated on K16 and passed `test_matmul`
  (`10/10`) plus `test_attention` (all three smoke shapes), but the focused
  benchmark did not improve LM-head forward versus the K32 default
  (`24870.98 us` TK versus `22183.82 us` cuBLASLt) and worsened several
  non-huge rows. TinyStories 3-step validation was effectively tied with the
  default at `2663.29 ms` total average, with steps `2660.03`, `2660.73`, and
  `2665.84 ms`, so the huge-N K tile stays at `32`.
- Rejected `LLMK_SM120_DWEIGHT_N128_K_TILE=8` at the compile gate after the
  decoupled huge-N K32 promotion. The N128 dWeight trait instantiated an
  8-row BF16 shared/register tile, and ThunderKittens rejected it with static
  assertions that rows and subtile rows must be divisible by the base tile row
  dimension, followed by zero-sized register-tile variables in
  `kernel_tn`. No smoke, benchmark, or TinyStories validation was run; the
  source default stays at `LLMK_SM120_DWEIGHT_N128_K_TILE=16`.
- Promoted decoupled SM120 huge-N forward K32 tiling while keeping N128
  dWeight on K16. A previous shared `LLMK_SM120_HUGE_N_K_TILE=32` test had
  improved LM-head forward but also changed dWeight N128 tiling; the new
  `LLMK_SM120_DWEIGHT_N128_K_TILE=16` default isolates dWeight from the
  forward huge-N K tile. The macro probe passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes), improved LM-head forward to
  `24804.11 us`, and validated at `2661.68 ms` total average over 3
  TinyStories steps. After promoting `LLMK_SM120_HUGE_N_K_TILE=32` as the
  source default, the no-override build passed `test_matmul` (`10/10`) and
  `test_attention`, benchmarked LM-head forward at `24882.34 us` versus
  `21984.87 us` cuBLASLt, and validated at `2663.76 ms` with steps `2657.78`,
  `2662.50`, and `2665.01 ms`. This improves the pure-TK default but still
  trails the cached cuBLASLt fallback near `2623 ms`.
- Rejected a scoped LM-head dWeight 128x64 K16 tile probe
  (`LLMK_SM120_DWEIGHT_LMHEAD_N64_K16=1`) after it improved the focused
  LM-head dWeight benchmark but regressed end-to-end training. The temporary
  route used a 128x64 K16 TN tile for large `OC x C` dWeight rows with
  `C == 768`, and the macro build passed `test_matmul` (`11/11`, including the
  guarded large-OC/C=768 row) plus `test_attention` (all three smoke shapes).
  The focused benchmark moved LM-head dWeight from the refreshed default
  `22997.94 us` to `22556.60 us`, but it still trailed cuBLASLt at
  `21171.98 us`. TinyStories 3-step validation regressed to `2684.95 ms` with
  steps `2680.53`, `2682.84`, and `2687.06 ms`, so the temporary alias,
  dispatch hook, and smoke row were removed.
- Promoted a scoped SM120 dInput direct B-column register-load route for the
  small-K GPT-2 qkv/attention-projection backward GEMMs
  (`LLMK_SM120_DINP_DIRECT_BCOL_SMALLK=1`). The route leaves fused dGELU,
  FC, and LM-head dInput on the existing row-load plus register-swap path.
  The pure-TK build passed `test_matmul` (`9/9`, including the new focused
  small-K dInput row) and `test_attention` (all three smoke shapes). Focused
  benchmarking improved qkv dInput to `1107.83 us` versus `1044.12 us`
  cuBLASLt (`1.06x`) and attention-projection dInput to `373.64 us` versus
  `384.36 us` cuBLASLt (`0.97x`). TinyStories 3-step validation averaged
  `2819.67 ms` with steps `2814.32`, `2819.12`, and `2825.58 ms`
  (`2822.35 ms` total average). This improves the current pure-TK source but
  remains just behind the supplied llm.c printed baseline and behind the
  cached cuBLASLt fallback, so the SM120 kernel-outperformance goal remains
  open.
- Rejected widening the same direct B-column dInput route to the GPT-2 FC
  backward shape with `LLMK_SM120_DINP_DIRECT_BCOL_K_CAP=3072`. The first
  `test_matmul` pass hit the known transient MLP-up forward row, the immediate
  rerun passed `10/10` including the medium-K dInput row, and `test_attention`
  passed all three smoke shapes. The focused benchmark made FC dInput
  competitive (`1561.27 us` TK versus `1582.27 us` cuBLASLt), but TinyStories
  3-step validation regressed to `2962.14 ms` with steps `2922.93`,
  `2951.67`, and `3011.83 ms` (`2981.75 ms` total average). The source default
  keeps the direct B-column cap at `2304`, leaving FC and LM-head dInput on the
  existing row-load path.
- Rejected a scoped direct B-column dInput swizzle override with
  `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=12`. The first `test_matmul` pass hit
  the known transient MLP-up forward row, the immediate rerun passed `9/9`,
  and `test_attention` passed all three smoke shapes. The focused benchmark
  made qkv and attention-projection dInput faster than cuBLASLt in that run
  (`1158.07 us` versus `1182.50 us`, and `383.48 us` versus `411.53 us`),
  but TinyStories 3-step validation regressed to `3019.29 ms` with steps
  `2979.29`, `3052.08`, and `3026.51 ms` (`3039.30 ms` total average). The
  direct B-column route continues to inherit the accepted
  `LLMK_SM120_DINP_SUPER_M=8` default.
- Rejected deferring the SM120 attention-projection dWeight split-K finish
  across attention backward. The candidate avoided the obvious `l_atty` data
  race by moving attention scratch to the already-dead MLP activation buffer,
  and passed `test_matmul` (`9/9`) plus `test_attention` (all three smoke
  shapes), but TinyStories 3-step validation raised memory use to
  `30977 MiB`, shifted the early loss/norm trace, and regressed to
  `2862.76 ms` with steps `2795.79`, `2848.28`, and `2944.22 ms`
  (`2896.25 ms` total average). The attproj backward path keeps the existing
  immediate dWeight finish before `l_atty` is reused by attention backward.
- Rebaselined the accepted small-K direct B-column dInput source after the
  rejected follow-ups. The pure-TK build passed `test_matmul` (`9/9`) and
  `test_attention` (all three smoke shapes), then TinyStories 3-step
  validation averaged `2679.28 ms` with steps `2679.01`, `2677.87`, and
  `2680.95 ms` (`2679.41 ms` total average). This beats the supplied llm.c
  printed baseline, but pure TK still trails the cached SM120 cuBLASLt fallback
  near `2623 ms`, so the kernel-outperformance goal remains open.
- Profiled the accepted small-K direct B-column dInput source with
  `LLMK_SM120_PROFILE_TRAIN_STEP=1`. The profiling build completed the required
  TinyStories 3-step validation with steps `2729.06`, `2721.95`, and
  `2729.35 ms` (`2725.65 ms` total average); the extra timing events account
  for the expected overhead versus the unprofiled `2679.28 ms` rebaseline. The
  dominant per-step buckets were forward (`~811 ms`), FC projection backward
  (`~344 ms`), FC backward (`~330 ms`), attention backward (`~283 ms`), QKV
  backward (`~241 ms`), and LM-head backward (`217-265 ms`). Source defaults
  were unchanged; the next optimization targets remain the GEMM-heavy forward,
  FC/FCProj backward, QKV/attention backward, and LM-head rows that still trail
  cuBLASLt.
- Added a disabled-by-default `LLMK_SM120_DINP_DIRECT_BCOL_LARGEK` probe for
  the LM-head-style large-K dInput route (`N == 768`, `K >= 8192`) plus a
  guarded smoke row. With the macro enabled, `test_matmul` passed `10/10`
  including the new large-K row and `test_attention` passed all three smoke
  shapes. The focused benchmark improved LM-head dInput to `22255.96 us`
  versus the accepted-source `23698.07 us`, but it still trailed cuBLASLt at
  `21038.20 us`. TinyStories 3-step validation improved to `2666.27 ms` with
  steps `2662.65`, `2663.99`, and `2668.56 ms`, faster than the accepted
  pure-TK rebaseline but still behind the cached cuBLASLt fallback near
  `2623 ms`. The macro remains off by default until a promotion run confirms
  it as the best source default.
- Promoted `LLMK_SM120_DINP_DIRECT_BCOL_LARGEK=1` as the default after a
  no-override source rebuild. The first `test_matmul` pass hit a transient
  accumulated dWeight row failure, the immediate rerun passed `10/10`, and
  `test_attention` passed all three smoke shapes. The focused benchmark kept
  LM-head dInput at `22242.20 us` versus `21053.73 us` cuBLASLt, and showed
  qkv/attention-projection dInput at or faster than cuBLASLt in that run.
  TinyStories 3-step validation averaged `2665.06 ms` with steps `2664.88`,
  `2664.34`, and `2665.77 ms`, improving the accepted pure-TK source while
  still leaving the cached cuBLASLt fallback as the training-time target to
  beat.
- Rejected a narrowly scoped FC projection dInput direct B-column probe
  (`N == 3072`, `K == 768`). The macro build passed `test_matmul` (`11/11`,
  including the guarded FCProj dInput row) and `test_attention` (all three
  smoke shapes), and the focused benchmark improved FCProj dInput to
  `1418.28 us` versus the prior promoted-source `1477.00 us`, but it still
  trailed cuBLASLt at `1365.63 us`. TinyStories 3-step validation regressed to
  `2669.49 ms` with steps `2665.80`, `2668.16`, and `2670.81 ms`, so the
  temporary FCProj direct route was removed.
- Rejected a fused-dGELU FC dInput direct B-column probe at the smoke gate.
  The temporary `LLMK_SM120_DINP_DIRECT_BCOL_DGELU_FC=1` alias targeted only
  the GPT-2 MLP-up backward shape (`N == 768`, `K == 3072`), but its guarded
  `test_matmul` row failed with max diff `0.7500` versus the `0.50` tolerance.
  No benchmark or TinyStories validation was run for the numerically unsafe
  candidate, and the temporary alias, dispatch hook, and smoke row were
  removed.
- Rejected a large-K-only direct B-column swizzle probe with
  `LLMK_SM120_DINP_DIRECT_BCOL_LARGEK_SUPER_M=16`. The macro build passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes), but
  the focused benchmark did not improve LM-head dInput (`22255.55 us` versus
  `21115.55 us` cuBLASLt) and worsened some unrelated rows. TinyStories
  3-step validation regressed to `2668.04 ms` with steps `2668.98`,
  `2666.50`, and `2669.59 ms`, so the temporary large-K swizzle alias was
  removed.
- Refreshed the SM120 per-step profile after promoting the large-K dInput
  direct B-column route. The `LLMK_SM120_PROFILE_TRAIN_STEP=1` build completed
  the required TinyStories 3-step validation with steps `2712.55`, `2711.10`,
  and `2713.60 ms` (`2712.35 ms` total average), with expected profiling
  overhead versus the unprofiled `2665.06 ms` default. The dominant buckets
  remained forward (`~813 ms`), FC projection backward (`~345 ms`), FC backward
  (`~329 ms`), attention backward (`~284 ms`), QKV backward (`~240 ms`), and
  LM-head backward (`~202-252 ms`), so the next optimization focus remains
  dense GEMM and attention-backward work rather than optimizer/update overhead.
- Rejected retesting `LLMK_SM120_DEFER_LMHEAD_DWEIGHT=0` after the large-K
  dInput promotion. The macro build passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  regressed to `2670.02 ms` with steps `2668.00`, `2669.25`, and
  `2670.79 ms`. The default keeps the LM-head dWeight side-stream deferral
  enabled because it remains faster than the immediate-wait path.
- Rejected retesting `LLMK_SM120_HUGE_N_K_TILE=32` on the current large-K
  dInput source. The macro build passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes), and the focused benchmark improved
  LM-head forward to `24791.84 us` versus the default-source `~25922 us`, but
  it worsened dWeight rows, including LM-head dWeight at `23698.38 us` versus
  `20872.77 us` cuBLASLt. TinyStories 3-step validation regressed to
  `2679.57 ms` with steps `2675.70`, `2678.06`, and `2681.09 ms`, so the
  huge-N/N128 K tile remains `16`.
- Rejected retesting current pure-TK codegen with `FORCE_NVCC_O=2` after the
  large-K dInput promotion. The O2 build passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes), but the focused benchmark worsened
  several rows, including qkv dInput (`1109.59 us` versus `1007.46 us`
  cuBLASLt) and attention-projection forward (`425.04 us` versus `371.09 us`).
  TinyStories 3-step validation averaged `2666.73 ms` with steps `2664.26`,
  `2665.24`, and `2668.21 ms`, slightly slower than the O3 source default.
- Rejected retesting qkv dWeight `LLMK_SM120_DWEIGHT_SPLIT_K=16` on the
  current large-K dInput source. The first `test_matmul` pass hit the known
  transient MLP-up forward row, the immediate rerun passed `10/10`, and
  `test_attention` passed all three smoke shapes. The focused benchmark
  improved qkv dWeight to `1165.80 us`, but qkv dInput and several other rows
  remained behind cuBLASLt. TinyStories 3-step validation regressed to
  `2669.86 ms` with steps `2667.80`, `2669.73`, and `2669.98 ms`, so qkv
  dWeight stays on the 8-way split-K default.
- Rejected a large-K dInput wide direct B-column probe that combined the
  promoted large-K direct-B-column route with the 256x64 dInput tile. The
  candidate build passed `test_matmul` (`11/11`, including the guarded wide
  large-K row) and `test_attention` (all three smoke shapes), but the focused
  benchmark worsened LM-head dInput to `25860.78 us` versus `21034.51 us`
  cuBLASLt. TinyStories 3-step validation regressed to `2682.03 ms` with steps
  `2677.86`, `2678.98`, and `2685.07 ms`, so the temporary wide direct
  B-column alias, dispatch hook, and smoke row were removed.
- Rejected a current-stack retest of `LLMK_SM120_HUGE_N_FORWARD_WIDE=1` after
  the large-K dInput promotion. The macro build passed `test_matmul` (`10/10`)
  and `test_attention` (all three smoke shapes), but the focused benchmark
  worsened LM-head forward to `26907.90 us` versus `22207.25 us` cuBLASLt.
  TinyStories 3-step validation regressed to `2684.10 ms` with steps
  `2678.84`, `2682.91`, and `2685.30 ms`, so the huge-N forward route remains
  on the 256x128 tile rather than the 256x64 wide tile.
- Rejected a forward-only `LLMK_SM120_HUGE_N_FORWARD_K_TILE=32` probe that
  tried to keep LM-head forward's earlier K32 improvement while leaving dWeight
  N128 on the accepted K16 route. The candidate failed the matmul smoke gate
  twice on the accumulated dWeight row (`3.6250` then `1.5938` max diff versus
  the `0.50` tolerance), so no benchmark or TinyStories validation was run and
  the temporary forward-only trait split was removed.
- Rejected a current-stack forward-only `LLMK_SM120_FORWARD_SUPER_M=8` swizzle
  split after the large-K dInput promotion. `test_matmul` passed on the third
  run after transient MLP-up and accumulated dWeight row failures, and
  `test_attention` passed all three smoke shapes. The focused benchmark
  improved FCProj forward (`1418.89 us` versus `1456.72 us` cuBLASLt) but
  worsened qkv and FC forward rows, and TinyStories 3-step validation regressed
  to `2668.91 ms` with steps `2666.62`, `2667.08`, and `2670.73 ms`. The
  temporary forward-only swizzle hook was removed.
- Rejected a current-stack `LLMK_SM120_DINP_SUPER_M=9` retest after the
  large-K direct B-column promotion. The macro build passed `test_matmul`
  (`10/10`) and `test_attention` (all three smoke shapes). The focused
  benchmark improved small-K dInput rows but worsened FC dInput and left
  LM-head dInput effectively unchanged; TinyStories 3-step validation regressed
  to `2668.82 ms` with steps `2667.20`, `2667.51`, and `2670.14 ms`, so the
  dInput swizzle remains `8`.
- Rejected scoped direct-B-column dInput swizzle
  `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=6`. The first `test_matmul` pass hit
  the known accumulated dWeight transient, the immediate rerun passed `10/10`,
  and `test_attention` passed all three smoke shapes. The focused benchmark
  worsened qkv dInput (`1069.06 us` versus `1007.53 us` cuBLASLt) while
  improving some FC/FCProj forward timings, and TinyStories 3-step validation
  regressed to `2668.37 ms` with steps `2665.33`, `2666.32`, and `2670.43 ms`.
  The direct-B-column route keeps inheriting `LLMK_SM120_DINP_SUPER_M=8`.
- Rejected adjacent direct-B-column dInput swizzle
  `LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M=7`. The macro build passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes), and
  the focused benchmark improved FC dInput to `1449.08 us` but did not improve
  small-K or LM-head dInput. TinyStories 3-step validation regressed to
  `2668.45 ms` with steps `2666.63`, `2667.55`, and `2669.35 ms`, so the
  direct-B-column swizzle remains at the inherited `8` default.
- Rejected retesting `LLMK_SM120_DWEIGHT_DIRECT_ACCUM=0` on the current
  large-K dInput source. The macro build passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes), but the focused benchmark kept
  accumulated dWeight rows behind cuBLASLt and worsened several rows. TinyStories
  3-step validation regressed to `2668.92 ms` with steps `2666.92`, `2666.94`,
  and `2670.89 ms`, so accumulated dWeight keeps the direct-accumulate default.
- Rejected qkv-only dWeight `LLMK_SM120_DWEIGHT_SPLIT_K=32` on the current
  large-K dInput source. The first `test_matmul` pass hit recurring transient
  MLP-up and accumulated dWeight row failures, the immediate rerun passed
  `10/10`, and `test_attention` passed all three smoke shapes. The focused
  benchmark improved qkv dWeight to `1194.99 us` (`1156.77 us` accumulated),
  but it still trailed cuBLASLt and TinyStories 3-step validation regressed to
  `2674.53 ms` with steps `2673.69`, `2673.01`, and `2676.05 ms`, so qkv
  dWeight remains on the 8-way split-K default.
- Rejected retesting `LLMK_SM120_ATTN_BWD_BLOCK=32` on the current large-K
  dInput source. The macro build passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  regressed to `2679.05 ms` with steps `2675.91`, `2676.53`, and `2681.57 ms`,
  so SM120 packed-QKV attention backward remains on the 16-row tile.
- Rejected a current-stack split-K dWeight partial-reducer block-size probe
  with `LLMK_SM120_DWEIGHT_REDUCE_BLOCK_SIZE=128`. The temporary hook passed
  `test_matmul` (`10/10`) and `test_attention` (all three smoke shapes), but
  the focused benchmark worsened qkv dWeight to `1292.81 us` (`1279.07 us`
  accumulated) and TinyStories 3-step validation regressed to `2669.82 ms`
  with steps `2667.56`, `2668.57`, and `2671.07 ms`. The reducer launch
  remains the fixed 256-thread default.
- Rejected retesting packed-QKV attention prep with
  `LLMK_SM120_DPREP_WARPS=4` on the current large-K dInput source. The macro
  build passed `test_matmul` (`10/10`) and `test_attention` (all three smoke
  shapes), but TinyStories 3-step validation regressed to `2669.68 ms` with
  steps `2666.71`, `2667.35`, and `2672.02 ms`, so dprep remains at the
  3-warp default.
- Refreshed the current-source SM120 pure-TK profile after the rejection
  rounds with `LLMK_SM120_PROFILE_TRAIN_STEP=1`. The profiling build completed
  the required TinyStories 3-step validation with steps `2717.12`, `2716.85`,
  and `2719.17 ms` (`2718.01 ms` total average). The dominant buckets remain
  forward (`~813-815 ms`), FC projection backward (`~345-347 ms`), FC backward
  (`~330 ms`), attention backward (`~283 ms`), QKV backward (`~238-242 ms`),
  LM-head backward (`~204-251 ms`), and final layernorm backward
  (`~132-178 ms`). Source defaults were unchanged; the next targets remain the
  GEMM-heavy forward, FC/FCProj backward, attention, QKV, and LM-head paths
  that still trail the cached cuBLASLt fallback.
- Refreshed the current-source focused SM120 matmul benchmark against its
  cuBLASLt variants to choose the next optimization target. Pure TK still wins
  qkv and attention-projection dInput (`0.96x` and `0.97x` TK/cuBLASLt) and
  FC projection forward (`0.96x`), but trails every dWeight row: qkv dW
  `1246.45 us` versus `988.08 us` (`1.26x`), attention-projection dW
  `484.88 us` versus `327.25 us` (`1.48x`), FC dW `1493.59 us` versus
  `1305.05 us` (`1.14x`), FC projection dW `1551.31 us` versus `1309.12 us`
  (`1.19x`), and LM-head dW `22997.94 us` versus `20981.42 us` (`1.10x`).
  LM-head forward remains a large gap (`25961.09 us` versus `22183.41 us`,
  `1.17x`). The next source work should prioritize TN/dWeight scheduling and
  the huge-N forward route.
- Rejected retesting global `LLMK_SM120_DWEIGHT_SUPER_M=3` on the current
  large-K dInput source. The macro build passed `test_matmul` (`10/10`) and
  `test_attention` (all three smoke shapes), but the focused benchmark was
  mixed and left material dWeight rows behind cuBLASLt: qkv dW `1247.25 us`
  versus `986.60 us`, attention-projection dW `491.61 us` versus `327.42 us`,
  FC dW `1516.47 us` versus `1307.88 us`, FC projection dW `1513.73 us`
  versus `1352.15 us`, and LM-head dW `22824.92 us` versus `20880.55 us`.
  TinyStories 3-step validation regressed to `2673.15 ms` with steps
  `2669.15`, `2671.66`, and `2674.64 ms`, so dWeight keeps the
  `LLMK_SM120_DWEIGHT_SUPER_M=2` default.
- Rejected a temporary huge-N forward 256x96x16 tile probe for the SM120
  LM-head forward path. The candidate compiled, but `test_matmul` failed at
  the GPT-2 LM-head smoke row with an illegal memory access before benchmarking
  or TinyStories validation. The temporary `LLMK_SM120_HUGE_N_FORWARD_N96`
  trait and aliases were removed; huge-N forward remains on the 256x128x16
  source default.
- Rejected a temporary SM120 LayerNorm backward block-size hook. A 256-thread
  `LLMK_SM120_LAYERNORM_BWD_BLOCK_SIZE` build compiled, but `test_layernorm`
  failed the backward `dbias` check (`1.510559` max diff versus `0.120`
  tolerance). A 1024-thread rebuild failed at launch with `invalid argument`.
  No TinyStories validation was run for either smoke-gate failure, and the
  temporary hook was removed; LayerNorm backward keeps the fixed 512-thread
  launch.
- Rejected a temporary forward-only huge-N K-tile split that tried to route
  the SM120 LM-head forward aliases through 256x128x32 while leaving dWeight
  N128 on the existing 128x128x16 path. The candidate compiled, but
  `test_matmul` failed the GPT-2 MLP-up forward row twice (`7.1562` and
  `7.6562` max diff versus `0.50` tolerance) before benchmarking or
  TinyStories validation. The temporary `LLMK_SM120_HUGE_N_FORWARD_K_TILE`
  hook was removed; huge-N forward and dWeight continue to share the K16
  source default.
- Rejected `LLMK_SM120_BIAS_BLOCK_SIZE=1024` for the CUDA bias-gradient
  reduction on the current large-K dInput source. The trainer build completed,
  but TinyStories 3-step validation regressed to `2669.44 ms` with steps
  `2665.44`, `2667.35`, and `2671.52 ms`, slower than the current pure-TK
  source default near `2665.06 ms`. Bias-gradient reduction keeps the
  512-thread source default.

## 2026-05-18 — SM120 RTX 5090 pure-TK rejection rounds

- Promoted cuBLASLt plan caching for the SM120 cuBLASLt fallback build. The
  macro probe with `LLMK_SM120_CACHE_CUBLASLT_PLANS=1` passed `test_matmul`
  (`8/8`) and `test_attention` (all three smoke shapes), then TinyStories
  3-step validation averaged `2624.24 ms` with steps `2624.77`, `2624.50`,
  and `2623.46 ms` (`2623.98 ms` excluding first-step warmup). Promoting the
  cache through the Makefile and rebuilding with no extra flags passed the same
  smoke gates and averaged `2623.44 ms` with steps `2623.20`, `2621.45`, and
  `2625.68 ms` (`2623.57 ms` excluding first-step warmup), improving the
  previous uncached cuBLASLt fallback baseline while leaving the pure-TK
  kernel-outperformance goal open.
- Tested the combined SM120 forward+dInput cuBLASLt fallback probe with
  dWeight still on TK. The build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), then TinyStories 3-step
  validation averaged `2708.81 ms` with steps `2703.57`, `2710.60`, and
  `2712.26 ms` (`2711.43 ms` excluding first-step warmup). The single-role
  effects stack, so forward plus dInput GEMMs account for most of the
  training-level gap; dWeight and residual scheduling/non-GEMM overhead remain
  the next blockers versus the all-cuBLASLt fallback.
- Tested the SM120 forward-only cuBLASLt fallback probe with dInput and
  dWeight still on TK. The build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), then TinyStories 3-step
  validation averaged `2771.90 ms` with steps `2769.95`, `2770.59`, and
  `2775.17 ms` (`2772.88 ms` excluding first-step warmup). This is effectively
  tied with the dInput-only fallback probe and shows forward and dInput GEMMs
  are both material SM120 blockers.
- Tested the SM120 dInput-only cuBLASLt fallback probe with forward and
  dWeight still on TK. The build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), then TinyStories 3-step
  validation averaged `2770.57 ms` with steps `2762.16`, `2774.51`, and
  `2775.03 ms` (`2774.77 ms` excluding first-step warmup). This is faster than
  the dWeight-only fallback probe, so backward dInput remains a higher-impact
  SM120 target than dWeight alone.
- Added disabled-by-default SM120 role-specific cuBLASLt probe switches for
  forward, dInput, and dWeight GEMMs, then tested the dWeight-only fallback
  with forward/dInput still on TK. The first `test_matmul` pass hit the known
  transient MLP-up forward row, the immediate rerun passed `8/8`, and
  `test_attention` passed all three smoke shapes. TinyStories 3-step
  validation averaged `2790.55 ms` with steps `2788.75`, `2793.55`, and
  `2789.35 ms` (`2791.45 ms` excluding first-step warmup), faster than the
  current pure-TK source but still well behind the all-cuBLASLt fallback, so
  dWeight is material but not the only remaining blocker.
- Rejected raising the current non-QKV dWeight split-K cap by building with
  `LLMK_SM120_DWEIGHT_SPLIT_K=16` and
  `LLMK_SM120_NON_QKV_DWEIGHT_SPLIT_K_CAP=16`. The macro build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but
  TinyStories 3-step validation averaged `2849.52 ms` with steps `2843.50`,
  `2851.88`, and `2853.17 ms` (`2852.52 ms` excluding first-step warmup),
  slower than the best accepted `LLMK_SM120_SUPER_M=7` run. The temporary cap
  macro was removed; source defaults remain qkv split-K 8 with non-QKV
  dWeight capped at 8-way split-K.
- Refreshed `bench_sm120_matmul` on the current pure SM120 TK source against
  the cuBLASLt fallback variants to pick the next optimization target. The
  largest remaining gaps were attention-projection dWeight (`517.94 us` TK vs
  `325.57 us` cuBLASLt, `1.59x`), attention-projection dInput (`456.36 us` vs
  `385.27 us`, `1.18x`), fcproj accumulated dWeight (`1663.72 us` vs
  `1363.95 us`, `1.22x`), and LM-head forward/dInput (`27100.01 us` vs
  `23148.19 us`, `1.17x`; `25118.81 us` vs `22283.56 us`, `1.13x`). No
  trainer timing was run for this benchmark-only round; source defaults were
  unchanged.
- Rejected deferring only QKV dWeight split-K finish until after LN1 backward
  with `LLMK_SM120_DEFER_QKV_DWEIGHT_FINISH=1`. The macro build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but
  TinyStories 3-step validation averaged `2843.11 ms` with steps `2836.81`,
  `2840.89`, and `2851.63 ms` (`2846.26 ms` excluding first-step warmup),
  slower than the best accepted `LLMK_SM120_SUPER_M=7` run. The temporary
  scheduling hook was removed.
- Rejected retesting `LLMK_SM120_DWEIGHT_SUPER_M=6` on the current source. The
  first `test_matmul` hit the known transient MLP-up forward row, the immediate
  rerun passed `8/8`, and `test_attention` passed all three smoke shapes, but
  TinyStories 3-step validation averaged `2842.01 ms` with steps `2836.51`,
  `2840.52`, and `2848.99 ms` (`2844.75 ms` excluding first-step warmup),
  slower than the best accepted `LLMK_SM120_SUPER_M=7` run. The dWeight swizzle
  remains `LLMK_SM120_DWEIGHT_SUPER_M=2`.
- Rejected disabling split-K dWeight/dInput overlap with
  `LLMK_SM120_OVERLAP_DINP_DWEIGHT=0`. The macro build passed `test_matmul`
  (`8/8`) and `test_attention` (all three smoke shapes), but TinyStories
  3-step validation regressed to `2892.90 ms` with steps `2871.76`,
  `2891.83`, and `2915.10 ms` (`2903.46 ms` excluding first-step warmup).
  The default keeps split-K dWeight side-stream overlap enabled.
- Rejected the focused older-fast combination `FORCE_NVCC_O=2`,
  `LLMK_SM120_SUPER_M=9`, and `LLMK_SM120_DWEIGHT_SPLIT_K=16` while keeping
  the current direct/overlap dWeight stack. The macro build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), then
  TinyStories 3-step validation averaged `2836.11 ms` with steps `2828.15`,
  `2841.45`, and `2838.72 ms` (`2840.08 ms` excluding first-step warmup).
  Promoting the same settings as the no-override source default also passed
  both smoke gates, but the required 3-step validation averaged `2842.13 ms`
  with steps `2835.39`, `2845.66`, and `2845.34 ms` (`2845.50 ms` excluding
  first-step warmup), slower than the best accepted `LLMK_SM120_SUPER_M=7`
  run. The source defaults remain O3, shared swizzle 7, and split-K 8.
- Rejected a current-source retest of the scoped 64x128 small-M dWeight TN
  tile route, `LLMK_SM120_DWEIGHT_N128_M64=1`, for `M <= 1024` and
  `N % 128 == 0` rows. The macro build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  regressed to `2862.49 ms` with steps `2854.01`, `2865.49`, and
  `2867.98 ms` (`2866.73 ms` excluding first-step warmup), slower than the
  best accepted `LLMK_SM120_SUPER_M=7` run. The temporary route was removed.
- Rejected extending LM-head dWeight deferral past final LNF by adding a
  compact LayerNorm/dbias scratch buffer and waiting only before the first
  attention-backward scratch reuse. The macro build passed `test_matmul`
  (`8/8`) and `test_attention` (all three smoke shapes), then TinyStories
  3-step validation averaged `2835.15 ms` with steps `2823.93`, `2835.22`,
  and `2846.30 ms` (`2840.76 ms` excluding first-step warmup). Promoting the
  same scheduling as the no-override source default also passed both smoke
  gates, but the required 3-step validation averaged `2841.42 ms` with steps
  `2838.20`, `2844.14`, and `2841.92 ms` (`2843.03 ms` excluding first-step
  warmup), slower than the best accepted `LLMK_SM120_SUPER_M=7` run. The
  temporary scratch/defer change was removed.
- Rejected a current-source retest of the temporary small-M dWeight swizzle
  split, `LLMK_SM120_DWEIGHT_SMALL_M_SUPER_M=3`. The hook routed TN dWeight
  rows with `M <= 1024` and `N % 128 == 0` through a separate 128x128 alias.
  The macro build passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), but TinyStories 3-step validation regressed to `2852.82 ms`
  with steps `2842.05`, `2854.78`, and `2861.64 ms` (`2858.21 ms` excluding
  first-step warmup), slower than the best accepted `LLMK_SM120_SUPER_M=7`
  run. The temporary hook was removed.
- Rejected a current-source forward-only `LLMK_SM120_FORWARD_SUPER_M=8` hook
  after a macro pass failed to reproduce as a promoted default. The macro build
  passed `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes),
  then TinyStories 3-step validation averaged `2835.17 ms` with steps
  `2827.83`, `2836.51`, and `2841.16 ms` (`2838.83 ms` excluding first-step
  warmup). Promoting the same value as the source default also passed both smoke
  gates, but the required no-override 3-step validation regressed to
  `2848.74 ms` with steps `2837.18`, `2852.21`, and `2856.82 ms`
  (`2854.52 ms` excluding first-step warmup), slower than the best accepted
  `LLMK_SM120_SUPER_M=7` run. The temporary hook was removed.
- Rejected `LLMK_SM120_ATTN_BWD_BLOCK=32` after retesting the attention
  backward block size on the current pure SM120 TK source. The macro build
  passed `test_attention` across all three smoke shapes, but TinyStories
  3-step validation averaged `2849.89 ms` with steps `2849.73`, `2854.41`,
  and `2845.54 ms` (`2849.97 ms` excluding first-step warmup), slower than the
  best accepted `LLMK_SM120_SUPER_M=7` run. The default remains block 16.
- Rejected `LLMK_SM120_DWEIGHT_SPLIT_K=4` after testing the lower split-K
  direction. The macro build passed `test_matmul` (`8/8`), but TinyStories
  3-step validation averaged `2847.76 ms` with steps `2844.02`, `2850.14`,
  and `2849.12 ms` (`2849.63 ms` excluding first-step warmup), slower than the
  best accepted `LLMK_SM120_SUPER_M=7` run. The default remains split-K 8.
- Re-ran the clean current pure SM120 TK default after the latest rejection
  rounds and disk cleanup. The default build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  averaged `2860.87 ms` with steps `2835.50`, `2871.28`, and `2875.82 ms`
  (`2873.55 ms` excluding first-step warmup), slower than the earlier accepted
  `LLMK_SM120_SUPER_M=7` macro run. Future candidates should compare against
  both the best observed `2837.54 ms` and this latest clean-source rebaseline.
- Rejected the bundled rollback toward the older O2/split-K dWeight stack:
  `FORCE_NVCC_O=2`, `LLMK_SM120_SUPER_M=9`,
  `LLMK_SM120_DWEIGHT_DIRECT_ACCUM=0`,
  `LLMK_SM120_OVERLAP_DINP_DWEIGHT=0`,
  `LLMK_SM120_DEFER_LMHEAD_DWEIGHT=0`,
  `LLMK_SM120_DWEIGHT_SPLIT_K=16`, and
  `LLMK_SM120_LARGE_DWEIGHT_SPLIT_K=8`. The build passed `test_matmul`
  (`8/8`), but TinyStories 3-step validation collapsed to `14350.65 ms` with
  steps `14394.69`, `14190.27`, and `14466.98 ms` (`14328.62 ms` excluding
  first-step warmup), so the current overlap/direct stack remains necessary.
- Rejected `LLMK_SM120_DWEIGHT_DIRECT_ACCUM=0` after restoring the scratch-plus-
  add accumulation path for direct dWeight rows. The macro build passed
  `test_matmul` (`8/8`), but TinyStories 3-step validation regressed from the
  current `2837.54 ms` source average to `2842.27 ms` with steps `2837.04`,
  `2840.61`, and `2849.17 ms` (`2844.89 ms` excluding first-step warmup). The
  direct-accumulate path remains the faster default.
- Rejected restoring split-K LM-head dWeight via
  `LLMK_SM120_DEFER_LMHEAD_DWEIGHT=0` plus
  `LLMK_SM120_LARGE_DWEIGHT_SPLIT_K=8` at the smoke gate. The build completed,
  but `test_matmul` failed the accumulated dWeight row twice, with max diffs
  `2.1094` and `1.7188` versus the `0.50` tolerance. No TinyStories 3-step
  timing was run because the candidate was numerically unsafe.
- Rejected the historical `FORCE_NVCC_O=2` plus `LLMK_SM120_SUPER_M=9`
  pairing at the matmul smoke gate on the current source. The macro build
  completed, but `test_matmul` failed the accumulated dWeight row first
  (`1.6250` versus `0.50`) and then failed the fused dGELU dInput row on rerun
  (`2.3789` versus `0.50`). No TinyStories 3-step timing was run because the
  candidate was numerically unsafe.
- Rejected reverting the current pure SM120 TK source to `FORCE_NVCC_O=2`.
  The O2 build passed `test_matmul` (`8/8`), but TinyStories 3-step validation
  averaged `2841.29 ms` with steps `2831.71`, `2841.73`, and `2850.44 ms`
  (`2846.08 ms` excluding first-step warmup), slower than the current O3
  `LLMK_SM120_SUPER_M=7` source average of `2837.54 ms`.
- Rejected `LLMK_SM120_DWEIGHT_N128=0` after retesting the older TN dWeight
  tile path. The macro build passed `test_matmul` (`8/8`), but TinyStories
  3-step validation regressed from the current `2837.54 ms` source average to
  `2957.47 ms` with steps `2927.65`, `2936.15`, and `3008.62 ms`
  (`2972.38 ms` excluding first-step warmup). The default keeps the 128x128
  dWeight tile enabled.
- Rejected the `-maxrregcount=128` build after a compiler occupancy retest. The
  first `test_matmul` run failed the accumulated dWeight row with max diff
  `1.5625` versus `0.50`, though two immediate reruns passed `8/8`. The
  TinyStories 3-step timing then regressed sharply from the current
  `2837.54 ms` source average to `3064.78 ms` with steps `3047.24`,
  `3061.16`, and `3085.94 ms` (`3073.55 ms` excluding first-step warmup).
- Rejected `LLMK_SM120_OVERLAP_DIRECT_DWEIGHT=0` after retesting whether the
  direct dWeight side-stream overlap still pays off. The macro build passed
  `test_matmul` (`8/8`), but TinyStories 3-step validation regressed from the
  current `2837.54 ms` source average to `2841.57 ms` with steps `2834.78`,
  `2846.59`, and `2843.34 ms` (`2844.97 ms` excluding first-step warmup). The
  default keeps direct dWeight overlap enabled.
- Rejected `LLMK_SM120_DINP_SUPER_M=9` after the adjacent dInput swizzle
  retest. The macro build passed `test_matmul` (`8/8`), but TinyStories
  3-step validation regressed from the current `2837.54 ms` source average to
  `2843.71 ms` with steps `2834.88`, `2845.46`, and `2850.79 ms`
  (`2848.12 ms` excluding first-step warmup). The default remains
  `LLMK_SM120_DINP_SUPER_M=8`.
- Rejected `LLMK_SM120_DINP_SUPER_M=7` at the matmul smoke gate. The macro
  build completed, but `test_matmul` failed reproducibly on the dInput A*B
  row with max diffs `7.6406` and `5.3906` versus the `0.50` tolerance. No
  TinyStories 3-step timing was run because the candidate was numerically
  unsafe.
- Rejected `LLMK_SM120_HUGE_N_K_TILE=32` after a huge-N LM-head retest. The
  macro build passed `test_matmul` (`8/8`), but TinyStories 3-step validation
  regressed from the current `2837.54 ms` source average to `2861.12 ms` with
  steps `2855.39`, `2862.02`, and `2865.94 ms` (`2863.98 ms` excluding
  first-step warmup). The default remains `LLMK_SM120_HUGE_N_K_TILE=16`.
- Rejected `LLMK_SM120_HUGE_N_M256=0` at the matmul smoke gate. The macro
  build completed, but `test_matmul` failed reproducibly: the first run failed
  the GPT-2 LM-head row with max diff `7.6719` versus `0.50`, and the rerun
  failed both MLP-up (`6.1406`) and LM-head (`6.9062`) forward rows. No
  TinyStories 3-step timing was run because the candidate was numerically
  unsafe.
- Rejected `LLMK_SM120_FORWARD_N96=0` after retesting the older forward tile
  route on the current stack. The macro build passed `test_matmul` (`8/8`) and
  completed the three timed TinyStories steps, but regressed from the current
  `2837.54 ms` source average to `2897.00 ms` with steps `2892.16`,
  `2893.89`, and `2904.95 ms`. The run then failed while writing the step-3
  checkpoint because the generated validation logs had filled the filesystem;
  old generated 5090 checkpoint payloads were removed and tracked marker files
  were restored before continuing. The default remains
  `LLMK_SM120_FORWARD_N96=1`.
- Rejected `LLMK_SM120_DWEIGHT_SPLIT_K=16` after a qkv-focused split-K
  retest. The macro build passed `test_matmul` (`8/8`), but TinyStories
  3-step validation regressed from the current `2837.54 ms` source average to
  `2840.11 ms` with steps `2836.83`, `2841.52`, and `2841.99 ms`
  (`2841.75 ms` excluding first-step warmup). The default remains
  `LLMK_SM120_DWEIGHT_SPLIT_K=8`.
- Rejected `LLMK_SM120_DWEIGHT_SUPER_M=1` at the matmul smoke gate. The macro
  build completed, but `test_matmul` failed reproducibly with changing unsafe
  rows: first the direct dWeight row hit max diff `1.7031` versus `0.50`, then
  the rerun failed the GPT-2 MLP-up forward row with max diff `7.1562`. No
  TinyStories 3-step timing was run because the candidate was numerically
  unsafe.
- Ran the current SM120 matmul microbenchmark against cuBLASLt for GPT-2 124M
  GEMM shapes. No TK shape beats cuBLASLt yet: qkv dWeight is `1.23x` slower,
  attproj dWeight is `1.54x` slower, fc fused forward is `1.06x` slower,
  fcproj dWeight is `1.16x` slower, and LM-head forward/dInput/dWeight remain
  `1.14x`/`1.13x`/`1.10x` slower with the largest absolute gap. This confirms
  that the next optimisation rounds need to focus on SM120 TK GEMM tiling,
  especially dWeight and LM-head, before the full trainer can beat the
  cuBLASLt fallback.
- Refreshed the SM120 cuBLASLt dense-GEMM fallback baseline under the current
  trainer stack. TinyStories 3-step validation averaged `2646.07 ms` with
  steps `2701.50`, `2615.67`, and `2621.05 ms` (`2618.36 ms` excluding
  first-step warmup), with finite loss/norm. This remains much faster than the
  current pure-TK `LLMK_SM120_SUPER_M=7` source average of `2837.54 ms`, so
  dense GEMM remains the required optimisation target before the SM120 path can
  beat all CUDA/cuBLASLt variants.
- Rejected the SM120 LayerNorm fallback comparison with
  `LLMK_DISABLE_TK_LAYERNORM` at the smoke gate. The macro build completed,
  but `test_layernorm` failed reproducibly on the backward `dbias` check with
  max diff `1.510559` versus the `0.120` tolerance, while forward, fused
  residual, `dinp`, and `dweight` checks passed. No TinyStories 3-step timing
  was run because the fallback build was numerically unsafe.
- Rejected the SM120 attention fallback comparison with
  `LLMK_DISABLE_TK_MHA_BWD` at the smoke gate. The macro build completed, but
  `test_attention` failed reproducibly on the `B=1 T=256 NH=2 HS=64` forward
  row with max diff `0.514648` versus the `0.080` tolerance, despite the
  `T=192` row passing. No TinyStories 3-step timing was run because the
  fallback build was numerically unsafe.
- Re-profiled the current `LLMK_SM120_SUPER_M=7` pure-TK source with
  `LLMK_SM120_PROFILE_TRAIN_STEP`. The profiled TinyStories 3-step run averaged
  `2879.36 ms` with expected profiler overhead. The dominant buckets remain
  dense-GEMM heavy: forward `~859 ms/step`, FC/FCProj backward `~704 ms/step`
  combined, attention backward `~297 ms/step`, QKV backward `~262 ms/step`,
  LM-head backward `~242 ms/step`, and final-LayerNorm overlap/wait
  `~170 ms/step`. The next optimisation focus therefore stays on the SM120 TK
  GEMM kernels rather than 1D reduction or optimizer paths.
- Rejected `LLMK_SM120_SUPER_M=6` after the adjacent swizzle retest. The
  macro build passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), but TinyStories 3-step validation regressed from the current
  `2837.54 ms` source average to `2847.42 ms` with steps `2841.31`,
  `2841.54`, and `2859.40 ms` (`2850.47 ms` excluding first-step warmup). The
  default remains `LLMK_SM120_SUPER_M=7`.
- Promoted `LLMK_SM120_SUPER_M=7` after the adjacent swizzle retest on the O3
  stack. The macro build passed `test_matmul` (`8/8`) and `test_attention`
  (all three smoke shapes), and TinyStories 3-step validation improved the
  current pure-TK source from `2840.38 ms` to `2837.54 ms` average with steps
  `2830.43`, `2833.28`, and `2848.90 ms` (`2841.09 ms` excluding first-step
  warmup). Pure TK still trails the supplied llm.c baseline and SM120 cuBLASLt
  fallback, so the goal remains open.
- Promoted `LLMK_SM120_SUPER_M=8` on top of the O3, dWeight-overlap, and
  narrowed LM-head wait stack. The macro build passed `test_matmul` (`8/8`)
  and `test_attention` (all three smoke shapes), and TinyStories 3-step
  validation improved the current pure-TK source from `2863.68 ms` to
  `2840.38 ms` average with steps `2831.43`, `2841.52`, and `2848.20 ms`
  (`2844.86 ms` excluding first-step warmup). Pure TK still trails the
  supplied llm.c baseline and SM120 cuBLASLt fallback, so the goal remains
  open.
- Promoted pure SM120 TK builds back to `FORCE_NVCC_O=3` after the latest
  dWeight-overlap and LM-head wait-point changes. The O3 build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), and
  TinyStories 3-step validation improved over the O2 source default from
  `2873.56 ms` to `2863.68 ms` average with steps `2880.65`, `2865.13`, and
  `2845.26 ms` (`2855.20 ms` excluding first-step warmup). Pure TK still
  trails the supplied llm.c baseline and SM120 cuBLASLt fallback, so the goal
  remains open.
- Promoted narrowing the deferred SM120 LM-head dWeight wait to immediately
  after final LayerNorm backward. This keeps the useful overlap between
  LM-head dWeight and LNF backward but avoids carrying the side-stream wait
  through the full transformer backward stack. The trainer build completed,
  and TinyStories 3-step validation improved from `2877.19 ms` to `2873.56 ms`
  average with steps `2863.32`, `2871.39`, and `2885.96 ms` (`2878.67 ms`
  excluding first-step warmup). Pure TK still trails the supplied llm.c
  baseline and SM120 cuBLASLt fallback, so the goal remains open.
- Rejected making the SM120 background dWeight streams low priority after the
  deferred LM-head wait change. The source hook passed `test_matmul` (`8/8`)
  and `test_attention` (all three smoke shapes), but TinyStories 3-step
  validation averaged `2877.38 ms` with steps `2873.22`, `2876.44`, and
  `2882.49 ms` (`2879.46 ms` excluding first-step warmup), slightly slower
  than the current default. The temporary priority hook was removed.
- Re-profiled after deferring the LM-head dWeight wait. The profiled
  TinyStories 3-step run averaged `2919.65 ms` with expected profiler overhead.
  `bwd_lmhead` dropped to `~263 ms/step`, but `bwd_lnf` rose to
  `~156 ms/step`, showing the deferred LM-head dWeight work now overlaps and
  competes with final-LayerNorm backward. The persistent dominant buckets are
  still forward `~872 ms/step`, FC/FCProj backward `~725 ms/step` combined,
  attention backward `~301 ms/step`, and QKV backward `~264 ms/step`.
- Promoted deferring the SM120 pure-TK LM-head dWeight wait until just before
  token-embedding backward. The GPT-2 backward path now starts the tied
  LM-head dWeight GEMM on a nonblocking side stream, runs the LM-head dInput
  and transformer-layer backward work on the main stream, then waits before
  `encoder_backward()` writes into `grads.wte`. `test_matmul` passed `8/8`,
  `test_attention` passed all three smoke shapes, and TinyStories 3-step
  validation improved the current source from `2880.34 ms` to `2877.19 ms`
  average with steps `2873.57`, `2873.77`, and `2884.22 ms` (`2879.00 ms`
  excluding the first-step warmup in the trainer's total-average line). Pure
  TK still trails the supplied llm.c baseline and SM120 cuBLASLt fallback, so
  the goal remains open.
- Re-profiled the current pure SM120 TK overlap stack with
  `LLMK_SM120_PROFILE_TRAIN_STEP`. The profiled TinyStories 3-step run averaged
  `2917.04 ms` with expected profiler overhead. The dominant buckets remain
  GEMM-heavy: forward `~871 ms/step`, LM-head backward `~429 ms/step`,
  FC/FCProj backward `~710 ms/step` combined, attention backward
  `~299 ms/step`, and QKV backward `~264 ms/step`; grad norm and update remain
  negligible. The next optimisation focus therefore stays on dense GEMM,
  especially LM-head and FC/FCProj backward.
- Rejected retesting `LLMK_SM120_LARGE_DWEIGHT_SPLIT_K=2` on top of the new
  dWeight overlap stack at the smoke gate. The macro build completed, but
  `test_matmul` failed the GPT-2 MLP-up forward row twice with max diffs
  `6.1465` and `6.5000` versus the `0.50` tolerance while the other seven rows
  passed. No TinyStories 3-step validation was run because the candidate was
  numerically unsafe.
- Promoted SM120 pure-TK direct dWeight overlap for dWeight rows where the
  split-K planner collapses to one part, notably the LM-head-sized dWeight
  route. These direct dWeight GEMMs now start on a nonblocking side stream,
  overlap with dInput and bias-grad on the main stream, then synchronize before
  `matmul_backward()` returns. `test_matmul` passed `8/8`, `test_attention`
  passed all three smoke shapes, and TinyStories 3-step validation improved
  the current overlap source from `2882.59 ms` to `2880.34 ms` average with
  steps `2873.48`, `2884.35`, and `2883.19 ms` (`2883.77 ms` excluding the
  first-step warmup in the trainer's total-average line). Pure TK still trails
  the supplied llm.c baseline and SM120 cuBLASLt fallback, so the goal remains
  open.
- Promoted SM120 pure-TK overlap between split-K dWeight partial kernels and
  independent work inside the same `matmul_backward()` call. The wrapper now starts
  eligible split-K dWeight work on the existing nonblocking part streams,
  launches dInput and bias-grad on the main stream, then waits and reduces the
  dWeight partials before returning. `test_matmul` passed `8/8`,
  `test_attention` passed all three smoke shapes, and TinyStories 3-step
  validation improved the current clean source from `2901.62 ms` to
  `2882.59 ms` average with steps `2879.04`, `2885.46`, and `2883.26 ms`
  (`2884.36 ms` excluding the first-step warmup in the trainer's total-average
  line). This is still slower than the supplied llm.c baseline and the SM120
  cuBLASLt fallback, so the next target remains the dense GEMM path, especially
  LM-head and dWeight rows.
- Rebaselined the clean pure SM120 TK source at `b27867f` after the latest
  rejection-only rounds. A no-extra-macro trainer build completed, and the
  TinyStories 3-step validation averaged `2901.62 ms` with steps `2892.39`,
  `2902.08`, and `2910.39 ms` (`2906.23 ms` excluding the first-step warmup in
  the trainer's total-average line). The run remained finite but still slower
  than the supplied llm.c baseline and the SM120 cuBLASLt fallback, so the
  optimisation goal remains open.
- Rejected macro-only `LLMK_SM120_BACKWARD_N96=1` at the smoke gate. The build
  completed for `test_matmul`, `test_attention`, `bench_sm120_matmul`, and
  `train_gpt2cu`, but `test_matmul` failed accumulated dWeight with max diff
  `1.5312` versus the `0.50` tolerance after passing the other seven rows. No
  focused benchmark or TinyStories 3-step validation was run because the
  candidate was numerically unsafe.
- Rejected deferring the final training-step loss synchronization from
  `gpt2_backward_and_reduce()` into the gradient-norm scalar copy. The trainer
  build passed with pure SM120 TK flags, but TinyStories 3-step validation
  regressed to `2907.85 ms` average with steps `2905.18`, `2905.01`, and
  `2913.37 ms` (`2909.19 ms` excluding the first-step warmup in the trainer's
  total-average line), so the source keeps the explicit final backward
  synchronization and mean-loss scaling.
- Rejected `LLMK_SM120_DWEIGHT_SPLIT_K_STREAMS=0`. The stream-disabled dWeight
  split-K build passed `test_attention`, and `test_matmul` passed on rerun after
  one transient GPT-2 MLP-up forward diff. The focused benchmark showed the
  change made dWeight far worse instead of closing the cuBLASLt gap: qkv dWeight
  `2541.68 us` vs cuBLASLt `1096.44 us`, attention-projection dWeight
  `2501.69 us` vs `367.10 us`, MLP dWeight `2519.91 us` vs `1512.63 us`,
  projection dWeight `2544.79 us` vs `1445.37 us`, and LM-head dWeight
  `26062.91 us` vs `23978.24 us`. The required TinyStories 3-step validation
  averaged `28770.46 ms` with steps `28484.18`, `28753.35`, and `28787.57 ms`,
  so the default concurrent split-K streams remain in place.
- Rejected macro-only `LLMK_SM120_DWEIGHT_SUPER_M=17`. The build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but the
  focused benchmark still left every dWeight row behind cuBLASLt: qkv dWeight
  `1298.92 us` versus `1074.27 us`, attention-projection dWeight `518.16 us`
  versus `346.52 us`, MLP dWeight `1765.66 us` versus `1414.57 us`, projection
  dWeight `1711.69 us` versus `1395.84 us`, and LM-head dWeight `25320.46 us`
  versus `23626.61 us`. TinyStories 3-step validation regressed to
  `37568.93 ms` with steps `36552.55`, `38345.41`, and `36792.45 ms`, so the
  source default remains `LLMK_SM120_DWEIGHT_SUPER_M=2`.
- Rejected macro-only `LLMK_SM120_DWEIGHT_SUPER_M=19` at the smoke gate. The
  first `test_matmul` run failed accumulated dWeight with max diff `2.3750`
  versus the `0.50` tolerance, and an immediate rerun failed the same
  accumulated dWeight row again while also hitting the recurring unrelated
  GPT-2 MLP-up forward transient. No focused benchmark or TinyStories 3-step
  validation was run because the dWeight kernel was numerically unsafe.
- Rejected macro-only `LLMK_SM120_DINP_SUPER_M=13` at the smoke gate. The
  first `test_matmul` run failed the plain dInput row with max diff `7.2109`
  versus the `0.50` tolerance, and an immediate rerun failed the same row with
  max diff `6.2969`. No focused benchmark or TinyStories 3-step validation was
  run because the dInput kernel was numerically unsafe.
- Rejected macro-only `LLMK_SM120_DINP_SUPER_M=14` at the smoke gate. The
  first `test_matmul` run failed plain dInput with max diff `8.2891` and also
  hit one dWeight row, then an immediate rerun again failed plain dInput with
  max diff `7.9375` versus the `0.50` tolerance. No focused benchmark or
  TinyStories 3-step validation was run because the dInput kernel was
  numerically unsafe.
- Rejected macro-only `LLMK_SM120_DINP_SUPER_M=15` at the smoke gate. The
  first `test_matmul` run failed plain dInput with max diff `7.2812`, and an
  immediate rerun failed the same row with max diff `7.5547` versus the `0.50`
  tolerance. No focused benchmark or TinyStories 3-step validation was run
  because the dInput kernel was numerically unsafe.
- Rejected raising `LLMK_SM120_HUGE_N_THRESHOLD` to `65536`, which routes the
  LM-head forward path away from the huge-N 256x128 tile. The build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), and the
  focused benchmark improved LM-head forward to `26623.37 us`, but it still
  trailed cuBLASLt (`23620.97 us`) and left the material dInput/dWeight rows
  behind, including attention-projection dWeight `585.34 us` versus
  `351.03 us`. TinyStories 3-step validation regressed to `3784.68 ms` with
  steps `3739.25`, `3787.55`, and `3781.81 ms`, so the threshold remains
  `8192`.
- Rejected pure SM120 TK codegen with `-Xptxas -maxrregcount=224`. The build
  passed `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes),
  but the focused benchmark still left every dWeight row behind cuBLASLt and
  only made attention-projection dInput faster (`402.32 us` versus
  `412.07 us`). TinyStories 3-step validation regressed badly to
  `21369.02 ms` with steps `21483.15`, `21252.05`, and `21485.99 ms`, so pure
  SM120 TK builds remain uncapped.
- Rejected a temporary large-M dWeight split-K hook that used 4-way split-K
  only for LM-head-sized dWeight rows. It passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), and the focused benchmark improved
  LM-head dWeight to `24154.74 us`, but that still trailed cuBLASLt
  (`22577.22 us`) and all material dWeight rows remained behind. TinyStories
  3-step validation regressed badly to `21070.57 ms` with steps `19565.49`,
  `21879.00`, and `20262.15 ms`, so the temporary hook was removed.
- Rejected `LLMK_SM120_DPREP_WARPS=6` for the packed-QKV attention backward
  prep helper. The build passed `test_matmul` (`8/8`) and `test_attention`
  (all three smoke shapes), but TinyStories 3-step validation regressed badly
  to `23199.38 ms` with steps `24328.04`, `23569.20`, and `22829.56 ms`, so
  the attention prep launch remains at `3` warps.
- Rejected `LLMK_SM120_DPREP_WARPS=7` for the same packed-QKV attention
  backward prep helper. The build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  regressed badly to `22620.45 ms` with steps `24170.88`, `22256.58`, and
  `22984.32 ms`, so the attention prep launch remains at `3` warps.
- Rejected `LLMK_SM120_DPREP_WARPS=8` for the same packed-QKV attention
  backward prep helper. The build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  still regressed to `15160.95 ms` with steps `15202.95`, `15265.78`, and
  `15056.12 ms`, so the attention prep launch remains at `3` warps.
- Rebaselined the current no-extra-macro pure SM120 TK source after the recent
  rejection-only rounds. `test_matmul` passed on the third run after transient
  dWeight and MLP-up failures, and `test_attention` passed all three smoke
  shapes. The focused benchmark still showed pure TK behind cuBLASLt on the
  material dInput/dWeight and LM-head rows, with attention-projection dWeight
  `535.35 us` versus cuBLASLt `346.98 us`. TinyStories 3-step validation was
  unexpectedly slow at `17955.85 ms` with steps `17470.85`, `18412.35`, and
  `17499.36 ms`; post-run GPU state was idle with no source diffs, so this is
  recorded as runtime evidence rather than a promoted source change.
- Rebaselined the current SM120 cuBLASLt fallback under the same runtime after
  the slow pure-TK rebaseline. `test_matmul` passed `8/8`, `test_attention`
  passed all three smoke shapes, and TinyStories 3-step validation averaged
  `2648.94 ms` with steps `2654.84`, `2646.18`, and `2651.71 ms`. This keeps
  the slowdown isolated to pure TK and confirms the fallback still beats the
  supplied llm.c baseline in the current runtime.
- Extended `bench_sm120_matmul` to time the accumulated dWeight (`dW+=`) path
  used by seven of the eight GPT-2 gradient-accumulation microsteps. The smoke
  gate hit one transient accumulated dWeight failure, then reran cleanly with
  `test_matmul` `8/8`; `test_attention` passed all three shapes. The new
  focused benchmark shows accumulated dWeight still behind cuBLASLt, especially
  attention-projection dW+= (`566.68 us` vs `386.60 us`) and qkv dW+=
  (`1394.36 us` vs `1158.64 us`). The required TinyStories 3-step validation
  averaged `26467.81 ms` with steps `26369.23`, `28045.54`, and `24890.08 ms`,
  so accumulated dWeight remains a target but does not alone explain the full
  pure-TK trainer slowdown.
- Added an opt-in `LLMK_SM120_PROFILE_TRAIN_STEP` trainer section profiler to
  identify the current pure-TK runtime sink without changing normal builds. The
  profiled build passed `test_matmul` (`8/8`) and `test_attention` (all three
  shapes). The required TinyStories 3-step validation averaged `19530.94 ms`
  with steps `20203.22`, `19179.90`, and `19881.97 ms`; the largest buckets
  were forward (`~4.8-5.2 s/step`), LM-head backward (`~5.1-5.2 s/step`), and
  packed attention backward (`~3.9-4.4 s/step`), while grad norm and update were
  negligible. This shifts the next optimization focus toward LM-head and
  packed-attention runtime paths.
- Ran the same section profiler on the SM120 cuBLASLt GEMM fallback. The build
  passed `test_matmul` (`8/8`) and `test_attention` (all three shapes), and the
  required TinyStories 3-step validation averaged `2997.91 ms` with steps
  `3007.18`, `2983.73`, and `3012.09 ms`. With the same attention kernels,
  forward dropped to `~0.80 s/step`, LM-head backward to `~0.38 s/step`, and
  packed attention backward to `~0.56 s/step`, so the pure-TK slowdown is a
  GEMM-path interaction rather than an attention-only kernel issue.
- Promoted `LLMK_SM120_LARGE_DWEIGHT_SPLIT_K=1` for the pure-TK trainer's
  LM-head-sized dWeight scratch allocation. Keeping qkv split-K at the source
  default while reducing only the large-dWeight scratch fanout cuts activation
  residency from `26030 MiB` to `25514 MiB` and removes the pathological
  multi-second pure-TK step regression. The source-default rebuild passed
  `test_matmul` (`8/8`) and `test_attention` (all three shapes), and the
  required TinyStories 3-step validation averaged `2962.54 ms` with steps
  `2939.85`, `3011.87`, and `2913.20 ms`. This is still behind the cuBLASLt
  fallback and the supplied llm.c baseline, so further pure-TK tuning remains
  required.
- Rejected promoting pure SM120 TK back to `FORCE_NVCC_O=3` after the reduced
  LM-head scratch default. The explicit O3 build passed `test_matmul` (`8/8`)
  and `test_attention` (all three shapes), and first averaged `2897.61 ms`, but
  the no-override source-default rebuild averaged `2973.54 ms` with steps
  `3014.38`, `3005.64`, and `2941.45 ms`. That did not beat the committed O2
  default or the supplied llm.c baseline, so pure SM120 TK remains on `O2`.
- Re-profiled the current pure SM120 TK default after reducing LM-head scratch.
  The profiled build passed `test_matmul` (`8/8`) and `test_attention` (all
  three shapes), and the required TinyStories 3-step validation averaged
  `2970.15 ms` with steps `2976.98`, `2964.26`, and `2976.04 ms`. The
  remaining gap versus the profiled cuBLASLt fallback is now mostly GEMM:
  forward is `~0.87 s/step`, LM-head backward `~0.43 s/step`, FC/FCProj
  backward `~0.74 s/step` combined, and QKV backward `~0.28 s/step`; packed
  attention backward is faster than the profiled fallback at `~0.30 s/step`.
- Promoted `LLMK_SM120_DWEIGHT_SPLIT_K=8` for the current reduced-scratch
  pure-TK path. The first source-default `test_matmul` hit the recurring
  accumulated dWeight transient, the immediate rerun passed `8/8`, and
  `test_attention` passed all three shapes. The required TinyStories 3-step
  validation averaged `2907.14 ms` with steps `2898.32`, `2905.02`, and
  `2909.25 ms`, improving the committed pure-TK default but still trailing the
  supplied llm.c baseline and the SM120 cuBLASLt fallback.
- Rejected `FORCE_NVCC_O=3` after the split-K=8 promotion. The O3 build passed
  `test_matmul` (`8/8`) and `test_attention` (all three shapes), but the
  required TinyStories 3-step validation averaged `2925.09 ms` with steps
  `2939.68`, `2916.43`, and `2933.76 ms`, slower than the O2 split-K=8 source
  default, so pure SM120 TK remains on `O2`.
- Rejected `LLMK_SM120_DWEIGHT_SUPER_M=3` after the split-K=8 promotion. The
  macro-only build passed `test_matmul` (`8/8`) and `test_attention` (all three
  shapes), but the required TinyStories 3-step validation averaged
  `2911.11 ms` with steps `2898.13`, `2909.78`, and `2912.44 ms`, slightly
  slower than the O2 split-K=8 source default, so the dWeight super-M remains
  `2`.
- Promoted direct SM120 TK accumulated-dWeight stores for cases where the
  split-K planner collapses to one part. The TN kernel now optionally loads the
  existing bf16 output tile and adds it in the float epilogue instead of writing
  a scratch GEMM followed by a separate add kernel. The build passed
  `test_matmul` (`8/8`) and `test_attention` (all three shapes). The focused
  benchmark improved several accumulated dWeight rows, including qkv dW+=
  `1297.88 us` and attention-projection dW+= `502.24 us`, but they still trail
  cuBLASLt (`1100.69 us` and `330.56 us`). TinyStories 3-step validation
  averaged `2895.13 ms` with steps `2892.79`, `2883.09`, and `2907.17 ms`,
  improving the pure-TK default while still trailing the supplied llm.c
  baseline and the SM120 cuBLASLt fallback.
- Rejected a temporary split-K dWeight direct-first hook that wrote split part
  0 directly to `dweight` and reduced only the remaining partials. The hook
  passed `test_matmul` (`8/8`) and `test_attention` (all three shapes), but the
  focused benchmark was mixed and TinyStories 3-step validation regressed to
  `2922.69 ms` with steps `2902.44`, `2919.40`, and `2925.98 ms`, so the hook
  was removed.
- Rejected retesting pure SM120 TK `FORCE_NVCC_O=3` after the direct
  accumulated-dWeight promotion. The O3 build passed `test_matmul` (`8/8`) and
  `test_attention` (all three shapes), but TinyStories 3-step validation
  averaged `2918.78 ms` with steps `2920.96`, `2913.55`, and `2924.01 ms`,
  slower than the O2 direct-accumulate default, so pure SM120 TK remains on
  `O2`.
- Re-profiled the current pure SM120 TK direct-accumulate default. The profiled
  TinyStories 3-step run averaged `2957.81 ms`; the dominant buckets remain
  GEMM-heavy: forward `~865 ms/step`, LM-head backward `~431 ms/step`,
  FC/FCProj backward `~733 ms/step` combined, attention backward
  `~294 ms/step`, and QKV backward `~285 ms/step`. The direct dWeight
  accumulate change did not move the next target away from forward and dense
  backward GEMM paths.
- Rejected `LLMK_SM120_LARGE_DWEIGHT_SPLIT_K=2` after the direct accumulated-
  dWeight promotion. The macro build passed `test_matmul` (`8/8`) and
  `test_attention` (all three shapes), but it raised activation allocation
  from `25514 MiB` to `25588 MiB` and TinyStories 3-step validation regressed
  to `2903.92 ms` with steps `2897.04`, `2899.60`, and `2908.24 ms`, so the
  large-dWeight split remains `1`.
- Rejected a temporary fused-GeLU forward huge-N threshold hook that routed the
  MLP-up `N=3072` shape through the huge-N tile. The candidate passed
  `test_matmul` (`8/8`) and `test_attention` (all three shapes), but the
  focused benchmark did not improve the `fc fwd+GeLU` row and TinyStories
  3-step validation regressed to `2929.27 ms` with steps `2922.14`, `2929.21`,
  and `2929.32 ms`, so the hook was removed.
- Rejected retesting `LLMK_SM120_SUPER_M=8` on the current direct-accumulate
  stack. The macro build passed `test_matmul` (`8/8`) and `test_attention`
  (all three shapes), and the focused benchmark improved qkv forward but
  worsened several dWeight rows. TinyStories 3-step validation averaged
  `2899.85 ms` with steps `2895.86`, `2895.82`, and `2903.89 ms`, slightly
  slower than the current source default, so the shared forward swizzle remains
  `9`.
- Rejected a temporary 512-thread fused-classifier block-size hook for SM120.
  The `test_fused_classifier` target still skips on sm_120, so the trainer was
  the runtime validation gate; TinyStories 3-step validation regressed to
  `2969.23 ms` with steps `2962.71`, `2964.04`, and `2974.43 ms`. The hook was
  removed and the classifier remains at the existing 256-thread block.
- Rejected a temporary forward-only `LLMK_SM120_FORWARD_SUPER_M=8` hook that
  left dInput and dWeight on the accepted swizzles. It passed `test_matmul`
  (`8/8`) and `test_attention` (all three shapes), but the focused benchmark
  worsened every forward row and TinyStories 3-step validation regressed to
  `2907.58 ms` with steps `2909.61`, `2903.73`, and `2911.44 ms`, so the hook
  was removed.

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
- Rejected a TN-only `LLMK_SM120_TN_INPLACE_LAYOUT_SWAP=0` hook for the pure
  SM120 TK dWeight kernel. It passed `test_matmul` (`8/8`) and `test_attention`
  (all three smoke shapes), and it slightly improved the direct LM-head dWeight
  row, but it worsened the smaller dWeight rows and the required TinyStories
  3-step validation regressed to `3616.17 ms` with steps `3586.78`, `3627.89`,
  and `3604.46 ms`. The dWeight kernel keeps the shared
  `LLMK_SM120_INPLACE_LAYOUT_SWAP=1` default.
- Rejected a scoped large-M TN no-inplace route for dWeight. The temporary
  hook kept the normal in-place layout swap for smaller dWeight rows and routed
  only large-M default-TN rows through the no-inplace alias. It passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but the
  focused benchmark remained mixed and the required TinyStories 3-step
  validation regressed to `3608.04 ms` with steps `3604.48`, `3615.18`, and
  `3600.90 ms`, so the hook was removed.
- Rejected reverting the pure SM120 TK huge-N forward route to the old 128x128
  tile (`LLMK_SM120_HUGE_N_M256=0`) under the O2 default. It passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but the
  focused benchmark regressed LM-head forward to `27372.12 us` versus cuBLASLt
  `21837.13 us`, and the required TinyStories 3-step validation averaged
  `3616.94 ms` with steps `3655.38`, `3628.57`, and `3605.31 ms`. The promoted
  256x128 huge-N tile remains the default.
- Rejected a rational-tanh approximation for the pure SM120 TK forward GELU
  epilogue. The temporary `LLMK_SM120_FORWARD_APPROX_GELU_TANH=1` hook passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but the
  focused benchmark worsened the fused FC forward row and the required
  TinyStories 3-step validation regressed to `3610.64 ms` with steps
  `3631.91`, `3613.16`, and `3608.12 ms` while shifting validation loss. The
  forward GELU epilogue keeps exact `tanhf`.
- Rejected `LLMK_SM120_HUGE_N_K_TILE=96` for the promoted pure SM120 TK
  256x128 huge-N forward route. The build failed at `test_matmul` compile time:
  ptxas reported the generated 256x128x96 huge-N kernel used `0x24000` bytes
  of shared data, above the SM120 `0x18c00` limit. The huge-N K tile remains
  `64`.
- Rejected a temporary 512x64 huge-N forward tile for pure SM120 TK. The
  `LLMK_SM120_HUGE_N_M512_N64=1` build failed at `test_matmul` compile time:
  ptxas reported the generated 512x64x64 huge-N kernel also used `0x24000`
  bytes of shared data, above the SM120 `0x18c00` limit. The huge-N tile
  remains 256x128x64.
- Rejected a 512x64x32 huge-N forward tile for pure SM120 TK. Lowering
  `LLMK_SM120_HUGE_N_K_TILE` to `32` let the candidate compile, but
  `test_matmul` aborted in ThunderKittens tensor-map setup because
  `st<bf16,512,32>` violates the `smem_shape[1] <= 256` assertion. The huge-N
  tile remains 256x128x64.
- Rejected a temporary 256x96x64 huge-N forward tile for pure SM120 TK. The
  `LLMK_SM120_HUGE_N_N96=1` build compiled, but `test_matmul` failed the
  LM-head forward row on the first run and failed both MLP-up and LM-head
  forward rows on the immediate rerun, so no benchmark or training validation
  was run. The huge-N tile remains 256x128x64.
- Rejected re-enabling the huge-N wide forward route
  (`LLMK_SM120_HUGE_N_FORWARD_WIDE=1`) under the current O2/M256 defaults. The
  build passed `test_matmul` (`8/8`) and `test_attention` (all three smoke
  shapes), but the focused benchmark regressed LM-head forward to
  `26982.29 us` versus cuBLASLt `21752.51 us`, and the required TinyStories
  3-step validation averaged `3595.56 ms` with steps `3609.62`, `3576.41`, and
  `3614.71 ms`. The huge-N wide route remains disabled.
- Rejected a temporary dWeight-only 256x96 TN tile for pure SM120 TK. The first
  `test_matmul` run hit the known transient MLP-up forward failure, but an
  immediate rerun passed `8/8` and `test_attention` passed all three smoke
  shapes. The focused benchmark improved qkv dWeight to `1360.28 us` but
  regressed attention-projection dWeight to `698.15 us` versus cuBLASLt
  `326.88 us`, worsened LM-head dWeight to `25951.42 us`, and the required
  TinyStories 3-step validation averaged `3585.04 ms` with steps `3597.60`,
  `3575.07`, and `3595.01 ms`. The temporary wide-N96 TN hook was removed.
- Rejected a temporary dWeight-only 192x64 TN tile for LM-head-style rows that
  are divisible by 192 but not 256. The first `test_matmul` pass again hit the
  unrelated transient MLP-up forward failure; an immediate rerun passed `8/8`
  and `test_attention` passed all three smoke shapes. The focused benchmark
  regressed LM-head dWeight to `29134.49 us` versus cuBLASLt `21018.38 us`,
  and TinyStories 3-step validation averaged `3618.86 ms` with steps
  `3612.57`, `3618.87`, and `3618.85 ms`. The temporary M192 hook was removed.
- Rejected a huge-N-only `SUPER_M=16` hook for the pure SM120 TK LM-head
  forward route. It passed `test_matmul` (`8/8`) and `test_attention` (all
  three smoke shapes), but the focused benchmark left LM-head forward unchanged
  at `24630.40 us` versus cuBLASLt `21826.90 us` and TinyStories 3-step
  validation averaged `3589.66 ms` with steps `3584.64`, `3608.17`, and
  `3571.15 ms`. The temporary huge-N `SUPER_M` hook was removed.
- Rejected creating the pure SM120 TK dWeight split-K part streams at high
  priority. The build passed `test_matmul` (`8/8`) and `test_attention` (all
  three smoke shapes), but the focused benchmark left dWeight ratios effectively
  unchanged and TinyStories 3-step validation averaged `3609.70 ms` with steps
  `3623.49`, `3618.42`, and `3600.99 ms`. The part streams remain normal
  nonblocking CUDA streams.
- Rejected a four-stream pool for the pure SM120 TK dWeight split-K part
  launches. It passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), but the focused benchmark worsened qkv and attention-projection
  dWeight while TinyStories 3-step validation still averaged `3601.50 ms` with
  steps `3554.16`, `3583.89`, and `3619.11 ms`. The split-K launcher continues
  to use one nonblocking stream per part.
- Rejected an 8-warp version of the non-wide 128x64 dWeight TN tile. It passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but the
  focused benchmark worsened LM-head dWeight to `26489.19 us` versus cuBLASLt
  `21165.43 us`, and TinyStories 3-step validation averaged `3621.02 ms` with
  steps `3619.64`, `3620.27`, and `3621.77 ms`. The non-wide dWeight TN path
  remains on the 4-warp 128x64 tile.
- Rejected a 128-thread launch for the SM120 split-K dWeight BF16 partial
  reducer. It passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), but the focused benchmark did not improve the dWeight rows
  and TinyStories 3-step validation averaged `3588.50 ms` with steps
  `3589.03`, `3602.48`, and `3574.52 ms`. The reducer remains at the
  256-thread launch.
- Rejected pure SM120 TK codegen with `-Xptxas -dlcm=ca`. It passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but the
  focused benchmark did not close the GEMM gap and TinyStories 3-step
  validation averaged `3591.41 ms` with steps `3608.48`, `3581.24`, and
  `3601.58 ms`. Pure SM120 TK builds keep the normal ptxas cache mode.
- Rejected pure SM120 TK `LLMK_SM120_DWEIGHT_SUPER_M=12`. The first
  `test_matmul` run hit the recurring unrelated MLP-up forward transient; an
  immediate rerun passed `8/8`, and `test_attention` passed all three smoke
  shapes. The focused benchmark still left all material dWeight rows behind
  cuBLASLt, and TinyStories 3-step validation averaged `3608.77 ms` with steps
  `3597.74`, `3599.97`, and `3617.57 ms`. The dWeight swizzle remains at the
  source default `2`.
- Rejected allowing 16-way split-K for large-row dWeight shapes while keeping
  smaller non-QKV rows capped at 8-way split-K. The first hook exposed an
  invalid-resource-handle bug because the side-stream/event arrays were still
  sized for the default 8 parts; after sizing those arrays for the larger test
  split count, the first `test_matmul` run hit the recurring unrelated MLP-up
  forward transient, the immediate rerun passed `8/8`, and `test_attention`
  passed all three smoke shapes. The focused benchmark worsened LM-head
  dWeight to `26333.16 us` versus cuBLASLt `21187.70 us`, and TinyStories
  3-step validation regressed to `4933.50 ms` with steps `4982.97`,
  `4955.35`, and `4911.65 ms`. The temporary large-row split-K hook was
  removed.
- Promoted `LLMK_SM120_SUPER_M=10` as the shared pure SM120 TK forward/dInput
  swizzle default after a macro-only A/B passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes). The focused benchmark remained
  behind cuBLASLt on most material GEMM rows, but the TinyStories 3-step
  validation improved sharply to `2828.57 ms` with steps `2825.73`, `2826.18`,
  and `2830.96 ms`. The no-override source-default rebuild passed the same
  smokes and averaged `2829.34 ms` with steps `2824.34`, `2827.24`, and
  `2831.44 ms`, bringing pure TK within noise of the supplied llm.c baseline
  while the kernel-outperformance goal remains open.
- Rejected the adjacent global `LLMK_SM120_SUPER_M=12` swizzle. The candidate
  compiled, but `test_matmul` failed the plain dInput row on two consecutive
  runs (`6.4219` and `6.8789` max diff versus `0.50` tolerance), so it was not
  benchmarked or validated with TinyStories training.
- Rejected the adjacent global `LLMK_SM120_SUPER_M=11` swizzle for the same
  reason. The candidate compiled, but `test_matmul` failed the plain dInput row
  on two consecutive runs (`5.7344` and `5.3125` max diff versus `0.50`
  tolerance), so it was not benchmarked or validated with TinyStories training.
- Promoted `LLMK_SM120_SUPER_M=9` as the shared pure SM120 TK forward/dInput
  swizzle default. The adjacent lower candidate passed `test_matmul` (`8/8`)
  and `test_attention` (all three smoke shapes); its focused benchmark was
  mixed versus `10`, but TinyStories 3-step validation improved slightly to
  `2826.59 ms` with steps `2822.01`, `2825.57`, and `2827.61 ms`. The
  no-override source-default rebuild passed the same smokes and averaged
  `2829.26 ms` with steps `2825.09`, `2826.07`, and `2832.45 ms`. Pure TK
  remains just above the supplied llm.c baseline and still behind cuBLASLt on
  the material dWeight rows.
- Split the SM120 dInput swizzle back out from the shared forward/huge-N
  swizzle by setting `LLMK_SM120_DINP_SUPER_M=10`. On top of the shared
  `LLMK_SM120_SUPER_M=9` default, the candidate passed `test_matmul` (`8/8`)
  and `test_attention` (all three smoke shapes). The focused benchmark was
  mixed but improved some dInput rows, and TinyStories 3-step validation
  averaged `2828.29 ms` with steps `2825.12`, `2826.14`, and `2830.43 ms`.
  The no-override source-default rebuild passed the same smokes and averaged
  `2829.00 ms` with steps `2823.58`, `2826.84`, and `2831.16 ms`.
- Rejected lowering the shared forward/huge-N swizzle to
  `LLMK_SM120_SUPER_M=7` while keeping dInput at `10`. The build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but the
  focused benchmark worsened the key forward rows and TinyStories 3-step
  validation regressed to `2829.93 ms` with steps `2827.15`, `2827.97`, and
  `2831.89 ms`.
- Rejected pure SM120 TK `LLMK_SM120_DWEIGHT_SUPER_M=10`. The build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but the
  focused benchmark still left every dWeight row well behind cuBLASLt and
  TinyStories 3-step validation regressed to `2834.88 ms` with steps
  `2831.16`, `2834.30`, and `2835.45 ms`.
- Rejected pure SM120 TK `LLMK_SM120_DWEIGHT_SUPER_M=14`. The build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but the
  focused benchmark still left the dWeight rows behind cuBLASLt and TinyStories
  3-step validation regressed to `2829.81 ms` with steps `2826.25`,
  `2827.27`, and `2832.35 ms`.
- Promoted `LLMK_SM120_DWEIGHT_SPLIT_K=16` under the current swizzle stack.
  The wrapper still caps non-QKV dWeight shapes at 8-way split-K, so this
  primarily restores qkv to 16 parts. The retest passed `test_matmul` (`8/8`)
  and `test_attention` (all three smoke shapes), improved qkv dWeight in the
  focused benchmark to `1395.13 us`, and improved TinyStories 3-step
  validation to `2820.16 ms` with steps `2818.54`, `2817.98`, and
  `2822.35 ms`. The no-override source-default rebuild passed the same smokes
  and averaged `2822.76 ms` with steps `2817.86`, `2821.16`, and `2824.36 ms`.
  Pure TK remains slightly above the supplied llm.c baseline and still behind
  cuBLASLt on the material dWeight rows.
- Rejected retesting `LLMK_SM120_DWEIGHT_SPLIT_K=32` under the current swizzle
  stack. The build passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), and qkv dWeight improved to `1357.94 us` in the focused
  benchmark, but the extra split launches regressed TinyStories 3-step
  validation to `2825.12 ms` with steps `2818.94`, `2823.61`, and
  `2826.63 ms`.
- Rejected a temporary 384-thread SM120 dWeight split-K partial reducer. The
  first `test_matmul` run hit the recurring unrelated MLP-up forward transient,
  the immediate rerun passed `8/8`, and `test_attention` passed all three smoke
  shapes. The focused benchmark was mixed and TinyStories 3-step validation
  regressed to `2824.67 ms` with steps `2821.31`, `2821.56`, and
  `2827.78 ms`, so the reducer remains at the 256-thread launch.
- Rejected retesting `LLMK_SM120_BIAS_BLOCK_SIZE=768` for the CUDA
  bias-gradient reduction under the current pure SM120 TK defaults. The macro
  build passed `test_matmul` (`8/8`) and `test_attention` (all three smoke
  shapes), and the first TinyStories 3-step validation averaged `2821.93 ms`
  with steps `2818.46`, `2820.37`, and `2823.49 ms`; however, the no-override
  source-default confirmation regressed to `2824.64 ms` with steps `2821.18`,
  `2822.14`, and `2827.14 ms`, so the source default stays at `512`.
- Rejected `LLMK_SM120_BIAS_BLOCK_SIZE=640` for the same CUDA bias-gradient
  reduction path. The build passed `test_matmul` (`8/8`) and `test_attention`
  (all three smoke shapes), but TinyStories 3-step validation regressed to
  `2827.01 ms` with steps `2819.71`, `2825.94`, and `2828.09 ms`.
- Rejected a scoped fixed-16 split-K partial reducer for the current qkv
  dWeight default. The source-default build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark did not
  improve qkv dWeight and TinyStories 3-step validation regressed to
  `2824.83 ms` with steps `2820.62`, `2823.10`, and `2826.57 ms`.
- Rejected retesting pure SM120 TK `FORCE_NVCC_O=3` under the current swizzle
  and split-K defaults. The O3 build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark was
  mixed and TinyStories 3-step validation regressed to `2826.49 ms` with steps
  `2821.64`, `2825.24`, and `2827.74 ms`, so pure SM120 TK builds keep the O2
  default.
- Rejected retesting `LLMK_SM120_DWEIGHT_SPLIT_K=64` under the current swizzle
  stack. The build passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), but the focused benchmark worsened qkv dWeight to
  `1458.18 us` versus cuBLASLt `988.77 us`, and TinyStories 3-step validation
  regressed to `2844.07 ms` with steps `2839.29`, `2842.58`, and
  `2845.57 ms`.
- Rejected enabling the SM120 backward `N % 96` tile
  (`LLMK_SM120_BACKWARD_N96=1`) under the current swizzle and split-K defaults.
  The build passed `test_matmul` (`8/8`) and `test_attention` (all three smoke
  shapes), and qkv dWeight improved to `1379.51 us` in the focused benchmark,
  but other backward rows regressed and TinyStories 3-step validation slowed to
  `2921.11 ms` with steps `2917.34`, `2920.17`, and `2922.04 ms`.
- Rejected disabling the fused SM120 dGELU dInput path
  (`LLMK_SM120_FUSE_DGELU=0`). The build passed `test_matmul` (`7/7`, with the
  fused dGELU smoke skipped by design) and `test_attention` (all three smoke
  shapes). The focused benchmark improved several plain backward GEMM timings,
  but TinyStories 3-step validation regressed to `2865.33 ms` with steps
  `2857.62`, `2862.82`, and `2867.85 ms`, so the source default remains fused.
- Rejected lowering the SM120 attention dprep launch to
  `LLMK_SM120_DPREP_WARPS=2`. The build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  regressed to `2827.45 ms` with steps `2819.23`, `2827.19`, and `2827.71 ms`.
  The source default remains the 4-warp dprep launch.
- Rejected increasing the SM120 attention backward tile to
  `LLMK_SM120_ATTN_BWD_BLOCK=32` under the current packed-QKV path. The build
  passed `test_attention` (all three smoke shapes), but TinyStories 3-step
  validation regressed to `2835.20 ms` with steps `2828.63`, `2833.57`, and
  `2836.83 ms`. The source default remains a 16-row backward tile.
- Rejected lowering the SM120 attention backward tile to
  `LLMK_SM120_ATTN_BWD_BLOCK=8`. The candidate does not compile: the SM120
  attention implementation explicitly supports only 16, 32, or 64 rows, and
  ThunderKittens register/shared tile types also require dimensions divisible
  by the tile granularity.
- Rejected compiling pure SM120 TK with `-Xptxas -maxrregcount=160`. The
  candidate repeatedly failed the `test_matmul` GPT-2 MLP-up forward case
  (`6.5625` then `6.8750` max diff versus `0.50` tolerance), so it was not
  benchmarked or validated with TinyStories training.
- Rejected `LLMK_SM120_BIAS_BLOCK_SIZE=576` for the CUDA bias-gradient
  reduction path. The build passed `test_matmul` (`8/8`) and `test_attention`
  (all three smoke shapes), but TinyStories 3-step validation regressed to
  `2826.59 ms` with steps `2820.46`, `2825.42`, and `2827.77 ms`. The source
  default remains `512`.
- Promoted an SM120 pure-TK dWeight TN 128x128 tile route
  (`LLMK_SM120_DWEIGHT_N128=1`) for supported dWeight shapes. The macro A/B
  passed `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes),
  improved several non-QKV dWeight rows in the focused benchmark, and averaged
  `2815.06 ms` on the TinyStories 3-step validation with steps `2807.96`,
  `2813.15`, and `2816.97 ms`. The no-override source-default rebuild passed
  the same smokes, reported qkv dWeight at `1371.03 us` in the focused
  benchmark, and confirmed `2815.36 ms` with steps `2810.13`, `2813.17`, and
  `2817.55 ms`. This beats the supplied llm.c 3-step average (`2818.23 ms`),
  but the kernel-outperformance goal remains open because the focused dWeight
  rows still trail cuBLASLt.
- Promoted `LLMK_SM120_DWEIGHT_SUPER_M=3` on top of the new dWeight 128x128
  TN route. The macro A/B passed `test_matmul` (`8/8`) and `test_attention`
  (all three smoke shapes), then improved TinyStories 3-step validation to
  `2812.57 ms` with steps `2808.11`, `2812.02`, and `2813.11 ms`. The
  no-override source-default rebuild passed the same smokes, stayed mixed in
  the focused benchmark, and confirmed an improved source-default trainer
  average of `2814.20 ms` with steps `2809.52`, `2811.78`, and `2816.62 ms`.
- Rejected adjacent `LLMK_SM120_DWEIGHT_SUPER_M=4` on the dWeight 128x128 TN
  route. The build passed `test_matmul` (`8/8`) and `test_attention` (all
  three smoke shapes), but the focused benchmark did not improve the dWeight
  rows and TinyStories 3-step validation averaged `2812.71 ms` with steps
  `2807.15`, `2811.48`, and `2813.94 ms`, slightly slower than the `3` macro
  retest.
- Rejected adjacent `LLMK_SM120_DWEIGHT_SUPER_M=5` on the dWeight 128x128 TN
  route. The build passed `test_matmul` (`8/8`) and `test_attention` (all
  three smoke shapes), but the focused benchmark remained mixed and TinyStories
  3-step validation averaged `2813.80 ms` with steps `2808.95`, `2812.76`, and
  `2814.83 ms`, so the source default stays at `3`.
- Rejected a temporary 256x128 8-warp TN tile route for supported SM120 dWeight
  shapes. The build passed `test_matmul` (`8/8`) and `test_attention` (all
  three smoke shapes), but the focused benchmark only improved the small
  attention-projection dWeight row while worsening larger dWeight rows, and
  TinyStories 3-step validation regressed to `2814.69 ms` with steps
  `2811.81`, `2812.75`, and `2816.63 ms`. The temporary wide-N128 hook was
  removed.
- Rejected scoping the dWeight 128x128 TN tile route to non-QKV shapes only.
  The build passed `test_matmul` (`8/8`) and `test_attention` (all three smoke
  shapes), and qkv dWeight improved in the focused benchmark when it fell back
  to the 256x64 path, but TinyStories 3-step validation regressed to
  `2815.63 ms` with steps `2812.28`, `2814.61`, and `2816.65 ms`. The
  temporary qkv scope knob was removed.
- Rejected raising the non-QKV dWeight split-K cap from 8 to 16 under the
  current 128x128 TN route. The build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark worsened
  the larger non-QKV dWeight rows and TinyStories 3-step validation regressed
  to `2822.17 ms` with steps `2818.06`, `2820.82`, and `2823.52 ms`. The
  temporary split-cap knob was removed.
- Rejected lowering the qkv dWeight split-K default to
  `LLMK_SM120_DWEIGHT_SPLIT_K=8` under the current 128x128 TN route. The build
  passed `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes),
  but qkv dWeight regressed to `1535.27 us` in the focused benchmark and
  TinyStories 3-step validation slowed to `2820.86 ms` with steps `2814.07`,
  `2817.38`, and `2824.34 ms`.
- Rejected enabling the SM90 TK LayerNorm forward/fused-residual path on SM120.
  Forward and fused-forward values passed the layernorm smoke tolerance, but
  `test_layernorm` repeatedly failed backward `dbias` (`0.496948` max diff
  versus `0.120` tolerance), including after splitting the SM120 forward opt-in
  away from the TK backward warp-reduction macro. The temporary SM120
  LayerNorm opt-in was removed and the CUDA fallback remains active.
- Rejected disabling the SM120 forward `N % 96` GEMM tile
  (`LLMK_SM120_FORWARD_N96=0`) under the current pure-TK defaults. The build
  passed `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes),
  but the focused benchmark badly regressed the qkv, attention-projection, MLP,
  and projection forward rows, and TinyStories 3-step validation slowed to
  `2874.11 ms` with steps `2868.20`, `2872.32`, and `2875.90 ms`.
- Promoted `LLMK_SM120_DPREP_WARPS=3` for the SM120 packed-QKV attention
  backward prep launch. The macro A/B passed `test_attention` (all three smoke
  shapes) and averaged `2812.93 ms` on TinyStories 3-step validation with
  steps `2808.03`, `2810.46`, and `2815.40 ms`. The no-override source-default
  rebuild passed `test_matmul` on rerun after the known MLP-up transient,
  passed `test_attention`, and confirmed `2813.17 ms` with steps `2807.86`,
  `2810.93`, and `2815.41 ms`.
- Rejected retesting adjacent `LLMK_SM120_DWEIGHT_SUPER_M=4` after promoting
  `LLMK_SM120_DPREP_WARPS=3`. The build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark stayed
  mixed and TinyStories 3-step validation averaged `2813.07 ms` with steps
  `2806.37`, `2812.08`, and `2814.05 ms`, not enough to displace the current
  `3` default.
- Rejected a temporary 128x192 4-warp TN tile route for SM120 dWeight. The
  build passed `test_matmul` (`8/8`) and `test_attention` (all three smoke
  shapes), but the focused benchmark catastrophically regressed every dWeight
  row (`attproj` dWeight `1272.32 us`, `lmhead` dWeight `45821.98 us`), and
  TinyStories 3-step validation slowed to `3294.73 ms` with steps `3286.69`,
  `3290.03`, and `3299.44 ms`. The temporary N192 tile/dispatch was removed.
- Promoted `LLMK_SM120_HUGE_N_K_TILE=32` for the SM120 huge-N/N128 tile family.
  Under the current dWeight N128 route this candidate passed `test_matmul`
  (`8/8`) and `test_attention` (all three smoke shapes), improved focused
  dWeight timings substantially (qkv `1220.39 us`, fc `1535.67 us`, fcproj
  `1556.57 us`, lmhead `23907.68 us`), and averaged `2749.99 ms` on the
  TinyStories 3-step validation with steps `2742.77`, `2746.69`, and
  `2753.29 ms`. The no-override source-default rebuild passed the same smokes,
  confirmed the benchmark improvement, and averaged `2748.57 ms` with steps
  `2743.59`, `2745.67`, and `2751.47 ms`.
- Rejected retesting adjacent `LLMK_SM120_DWEIGHT_SUPER_M=4` after promoting
  the huge-N/N128 K tile to `32`. The macro build passed `test_matmul` (`8/8`
  on the immediate rerun after the known transient MLP-up row) and
  `test_attention` (all three smoke shapes), but the focused benchmark stayed
  mixed and the TinyStories 3-step validation averaged `2750.21 ms` with steps
  `2743.93`, `2748.14`, and `2752.27 ms`, slower than the committed
  `LLMK_SM120_DWEIGHT_SUPER_M=3` source default.
- Rejected retesting adjacent `LLMK_SM120_DWEIGHT_SUPER_M=2` after promoting
  the huge-N/N128 K tile to `32`. The macro build passed `test_matmul` (`8/8`)
  and `test_attention` (all three smoke shapes), and the focused benchmark
  improved LM-head dWeight to `23525.40 us`, but smaller dWeight rows remained
  mixed and TinyStories 3-step validation averaged `2749.98 ms` with steps
  `2742.78`, `2748.08`, and `2751.88 ms`, still slower than the committed
  `LLMK_SM120_DWEIGHT_SUPER_M=3` source default.
- Rejected retesting `LLMK_SM120_DWEIGHT_SPLIT_K=32` after promoting the
  huge-N/N128 K tile to `32`. The macro build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), and the focused benchmark nudged
  qkv dWeight down to `1205.09 us`, but FC dWeight regressed to `1566.62 us`
  and TinyStories 3-step validation averaged `2750.73 ms` with steps
  `2745.21`, `2749.58`, and `2751.88 ms`, slower than the committed
  split-K `16` default.
- Promoted `LLMK_SM120_HUGE_N_K_TILE=16` for the shared SM120 huge-N/N128 tile
  family. The macro A/B build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), improved dWeight rows despite a
  slower LM-head forward path (qkv dWeight `1182.34 us`, attproj `504.47 us`,
  fcproj `1517.71 us`, lmhead `22671.24 us`), and averaged `2746.71 ms` on
  TinyStories with steps `2739.70`, `2745.03`, and `2748.40 ms`. The
  no-override source-default rebuild passed the same smokes, confirmed the
  dWeight improvement (qkv `1177.66 us`, attproj `497.04 us`, fcproj
  `1507.44 us`, lmhead `22761.23 us`), and averaged `2745.01 ms` with steps
  `2739.79`, `2742.41`, and `2747.62 ms`.
- Promoted `LLMK_SM120_DWEIGHT_SUPER_M=2` on top of the K-tile 16 route. The
  macro A/B build passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), kept dWeight rows close to the K-tile 16 default (qkv
  `1176.31 us`, attproj `496.01 us`, fcproj `1504.80 us`, lmhead
  `22917.54 us`), and averaged `2742.78 ms` on TinyStories with steps
  `2740.98`, `2741.63`, and `2743.93 ms`. The no-override source-default
  rebuild passed the same smokes, benchmarked qkv/attproj/lmhead dWeight at
  `1173.46`, `484.78`, and `22678.40 us`, and averaged `2744.85 ms` with
  steps `2740.57`, `2742.70`, and `2747.01 ms`, narrowly ahead of the prior
  K-tile 16 source default.
- Rejected retesting `LLMK_SM120_DWEIGHT_SUPER_M=1` on top of the K-tile 16
  route. Unlike earlier stacks it now passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but focused benchmark rows were
  mixed (qkv dWeight `1182.84 us`, attproj `491.18 us`, lmhead `22865.17 us`)
  and TinyStories 3-step validation averaged `2746.08 ms` with steps
  `2741.13`, `2743.56`, and `2748.60 ms`, slower than the committed
  `LLMK_SM120_DWEIGHT_SUPER_M=2` default.
- Rejected retesting `LLMK_SM120_DWEIGHT_SUPER_M=4` on top of the K-tile 16
  route. The macro build passed `test_matmul` (`8/8`) and `test_attention`
  (all three smoke shapes), but the focused benchmark worsened the material
  qkv/attproj/lmhead dWeight rows versus the committed `2` default (qkv
  `1183.03 us`, attproj `504.33 us`, lmhead `23018.25 us`). TinyStories
  3-step validation averaged `2744.67 ms` with steps `2739.28`, `2742.32`,
  and `2747.01 ms`, too small a timing difference to justify worse kernel
  evidence.
- Rejected retesting `LLMK_SM120_DWEIGHT_SPLIT_K=32` on top of the K-tile 16
  and dWeight `SUPER_M=2` defaults. The macro build passed `test_matmul`
  (`8/8`) and `test_attention` (all three smoke shapes), and improved qkv
  dWeight to `1156.05 us`, but it worsened attproj forward and LM-head dWeight
  versus the committed split-K `16` default. TinyStories 3-step validation
  averaged `2746.83 ms` with steps `2741.67`, `2743.91`, and `2749.76 ms`, so
  the qkv-only split-K expansion stays rejected.
- Promoted `LLMK_SM120_DINP_SUPER_M=8` for the SM120 dInput swizzle on top of
  the K-tile 16 and dWeight `SUPER_M=2` stack. The macro A/B build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), had a
  mixed dInput benchmark (qkv `1109.62 us`, fcproj `1476.74 us`, lmhead
  `23447.24 us`), and averaged `2742.72 ms` on TinyStories with steps
  `2736.41`, `2739.23`, and `2746.21 ms`. The no-override source-default
  rebuild passed the same smokes, benchmarked dInput rows at qkv `1086.82 us`,
  fcproj `1475.79 us`, and lmhead `23621.88 us`, and averaged `2743.37 ms`
  with steps `2741.44`, `2740.31`, and `2746.44 ms`.
- Rejected `LLMK_SM120_DINP_SUPER_M=6` on top of the current K-tile 16 stack.
  The macro build compiled, but `test_matmul` failed the plain dInput row twice
  (`5.2812` then `7.9258` max abs diff versus tolerance `0.50`), so no focused
  benchmark or TinyStories training validation was run for this incorrect
  candidate.
- Rejected `LLMK_SM120_DINP_SUPER_M=7` on top of the current K-tile 16 stack.
  The macro build compiled, but `test_matmul` failed the plain dInput row twice
  (`6.8594` then `6.7344` max abs diff versus tolerance `0.50`), so no focused
  benchmark or TinyStories training validation was run for this incorrect
  candidate.
- Rejected `LLMK_SM120_DINP_SUPER_M=9` on top of the current K-tile 16 stack.
  The macro build passed `test_matmul` (`8/8` on rerun after the known
  transient MLP-up row) and `test_attention` (all three smoke shapes), but the
  focused benchmark was mixed and TinyStories 3-step validation averaged
  `2743.50 ms` with steps `2736.84`, `2741.46`, and `2745.55 ms`, slightly
  slower than the committed `LLMK_SM120_DINP_SUPER_M=8` default.
- Rejected disabling the 256x128 huge-N forward tile
  (`LLMK_SM120_HUGE_N_M256=0`) on top of the K-tile 16 stack. The macro build
  passed `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes)
  and improved LM-head forward to `25171.34 us`, but it worsened other
  benchmark rows and TinyStories 3-step validation regressed to `2748.11 ms`
  with steps `2746.77`, `2744.09`, and `2752.13 ms`. The source default keeps
  the 256x128 huge-N forward tile.
- Rejected `LLMK_SM120_HUGE_N_K_TILE=8`. The macro build failed at
  `test_matmul` compile time because the TK shared/register tile types require
  BF16 dimensions divisible by their base tile dimensions; ptxas reported
  repeated `Cols must be divisible by the tile dimension`, `Rows must be
  divisible by the tile dimension`, and zero-sized register tile errors. No
  benchmark or TinyStories validation was run for this invalid tile.
- Rejected retesting pure SM120 TK `FORCE_NVCC_O=3` on top of the current
  K-tile 16/dInput stack. The explicit O3 build passed `test_matmul` (`8/8`)
  and `test_attention` (all three smoke shapes), had mixed focused benchmark
  results, and averaged `2742.35 ms` on TinyStories with steps `2736.77`,
  `2741.25`, and `2743.44 ms`. After temporarily changing the Makefile default,
  the no-override source-default rebuild passed the same smokes but averaged
  `2743.67 ms` with steps `2737.96`, `2740.97`, and `2746.37 ms`, slower than
  the committed O2 source default, so pure SM120 TK keeps `FORCE_NVCC_O=2`.
- Rejected a temporary `LLMK_SM120_DWEIGHT_WIDE_N128=1` hook that routed the
  SM120 N128 TN dWeight path through the 256x128 tile. The candidate passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but
  `bench_sm120_matmul` aborted before timing because the 256-row TN tile
  asserts `M % T::M_TILE == 0`, which is not true for every covered dWeight
  row. The temporary hook was removed and no TinyStories validation was run.
- Rejected increasing SM120 packed-QKV attention prep to
  `LLMK_SM120_DPREP_WARPS=4` on top of the current K-tile 16 stack. The macro
  build passed `test_matmul` (`8/8`) and `test_attention` (all three smoke
  shapes), but TinyStories 3-step validation averaged `2744.14 ms` with steps
  `2739.37`, `2742.82`, and `2745.46 ms`, slower than the committed
  `LLMK_SM120_DPREP_WARPS=3` default.
- Rejected increasing SM120 attention forward tiling to
  `LLMK_SM120_ATTN_FWD_BLOCK=64`. The macro build passed `test_matmul` (`8/8`)
  and `test_attention` (all three smoke shapes), but TinyStories 3-step
  validation regressed sharply to `3104.70 ms` with steps `3104.32`, `3099.08`,
  and `3110.32 ms`. The forward attention tile remains `32`.
- Rejected intermediate `LLMK_SM120_ATTN_BWD_BLOCK=24`. The build failed at
  `test_attention` compile time because the SM120 attention backward kernel
  explicitly supports only `16`, `32`, or `64`, and TK register/shared tile
  types also require tile dimensions divisible by their base dimensions. No
  runtime validation was run for this invalid tile.
- Rejected `LLMK_SM120_ATTN_BWD_BLOCK=64`. The macro build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but
  TinyStories 3-step validation regressed catastrophically to `7223.85 ms`
  with steps `7309.41`, `7221.44`, and `7226.25 ms`. The backward attention
  tile remains `16`.
- Rejected the SM120 attention `LLMK_SM120_ATOMIC_DQ` accumulation path. The
  macro build passed `test_matmul` (`8/8` on rerun after the known transient
  MLP-up row) and `test_attention`, but TinyStories 3-step validation regressed
  to `3111.17 ms` with steps `3105.52`, `3108.17`, and `3114.17 ms`. The
  default non-atomic dQ path remains in place.
- Rejected retesting the adjacent shared forward/huge-N swizzle
  `LLMK_SM120_SUPER_M=8` on top of the current K-tile 16, dWeight
  `SUPER_M=2`, and dInput `SUPER_M=8` stack. The macro build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), and
  the focused benchmark improved qkv forward to `1073.78 us` versus cuBLASLt
  `1109.21 us`, but dInput and dWeight rows still trailed cuBLASLt and
  TinyStories 3-step validation regressed to `2747.90 ms` with steps
  `2742.96`, `2745.97`, and `2749.82 ms`. The shared SM120 swizzle remains at
  the source default `9`.
- Rejected retesting the adjacent shared forward/huge-N swizzle
  `LLMK_SM120_SUPER_M=10` on the same current stack. The macro build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but the
  focused benchmark worsened key qkv and attention dWeight rows versus the
  committed default (qkv dWeight `1271.68 us`, attproj dWeight `518.32 us`) and
  TinyStories 3-step validation regressed to `2748.25 ms` with steps
  `2741.36`, `2746.30`, and `2750.21 ms`. The shared SM120 swizzle remains at
  `9`.
- Rejected a temporary `LLMK_SM120_HUGE_N_SUPER_M` hook that split the LM-head
  huge-N forward/dInput/dWeight swizzle from the shared SM120 swizzle. The
  `LLMK_SM120_HUGE_N_SUPER_M=8` build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), and the focused benchmark slightly
  improved some forward rows such as fcproj forward (`1420.06 us` versus
  cuBLASLt `1432.39 us`), but LM-head and the material dInput/dWeight rows
  still trailed cuBLASLt. TinyStories 3-step validation regressed to
  `2747.36 ms` with steps `2738.78`, `2745.63`, and `2749.09 ms`, so the
  temporary hook was removed and huge-N aliases again use the shared
  `LLMK_SM120_SUPER_M=9`.
- Rejected disabling the SM120 dWeight 128x128 TN route
  (`LLMK_SM120_DWEIGHT_N128=0`) under the current K-tile 16/dInput stack. The
  macro build passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), but the focused benchmark regressed every material dWeight row
  (qkv `1390.10 us`, attproj `553.96 us`, fcproj `1720.52 us`, lmhead
  `25678.11 us`) and TinyStories 3-step validation slowed to `2836.51 ms` with
  steps `2829.63`, `2833.90`, and `2839.13 ms`. The dWeight N128 route remains
  enabled.
- Rejected the missing adjacent dWeight swizzle retest
  `LLMK_SM120_DWEIGHT_SUPER_M=3` on the current K-tile 16/dInput stack. The
  macro build passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), but the focused benchmark stayed behind cuBLASLt on every
  dWeight row and worsened the qkv row to `1178.73 us`. TinyStories 3-step
  validation regressed to `2749.03 ms` with steps `2741.59`, `2745.81`, and
  `2752.25 ms`, so the source default remains `LLMK_SM120_DWEIGHT_SUPER_M=2`.
- Rejected retesting `LLMK_SM120_DINP_SUPER_M=10` under the current K-tile
  16/dWeight stack. The macro build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark
  worsened the material dInput rows versus the committed `8` default (qkv
  `1134.85 us`, fc `1502.11 us`, lmhead `23720.62 us`) and TinyStories
  3-step validation regressed to `2747.71 ms` with steps `2742.78`, `2744.83`,
  and `2750.59 ms`. The dInput swizzle remains `8`.
- Rejected retesting `LLMK_SM120_DINP_SUPER_M=5` under the current K-tile
  16/dWeight stack. The macro build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark did not
  improve the material dInput rows and worsened the fused FC forward timing to
  `1579.54 us`. TinyStories 3-step validation regressed to `2747.68 ms` with
  steps `2743.72`, `2747.19`, and `2748.17 ms`, so the dInput swizzle remains
  `8`.
- Rejected a partial K-loop unroll in all SM120 GEMM kernels. Changing the
  three K loops from `#pragma unroll 1` to `#pragma unroll 2` compiled, but
  `test_matmul` failed reproducibly on two runs: the GPT-2 MLP-up forward row
  reported max diffs `7.7031` then `7.7188`, and the fused dGELU row reported
  `14.6562` then `14.2344`, all versus the `0.50` tolerance. The temporary
  source edit was reverted and no benchmark or TinyStories validation was run.
- Rejected isolating that partial unroll to the SM120 dWeight TN kernel only.
  The `kernel_tn`-only `#pragma unroll 2` edit passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark
  regressed dWeight rows and several unrelated timings, including qkv dWeight
  `1338.20 us` and LM-head dWeight `25744.22 us`. TinyStories 3-step
  validation slowed to `2897.79 ms` with steps `2861.16`, `2900.22`, and
  `2895.35 ms`, so the source edit was reverted.
- Rejected isolating the stricter cp.async wait policy to the SM120 dWeight TN
  kernel only. Changing `kernel_tn` from `load_async_wait<2>` to
  `load_async_wait<1>` passed `test_matmul` (`8/8`) and `test_attention` (all
  three smoke shapes), but the focused benchmark regressed every dWeight row
  badly (qkv `1585.53 us`, attproj `590.76 us`, lmhead `27897.90 us`) and
  TinyStories 3-step validation slowed to `2947.35 ms` with steps `2949.62`,
  `2947.36`, and `2947.33 ms`. The temporary source edit was reverted.
- Rejected retesting the SM120 backward 128x96 tile route
  (`LLMK_SM120_BACKWARD_N96=1`) on the current K-tile 16/dInput stack. The
  macro build passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), but the focused benchmark severely regressed dInput rows (qkv
  `1342.93 us`, attproj `493.00 us`, lmhead `28945.54 us`) and TinyStories
  3-step validation slowed to `2901.21 ms` with steps `2878.86`, `2896.43`,
  and `2906.00 ms`. The route remains disabled.
- Rejected `LLMK_SM120_BIAS_BLOCK_SIZE=384` for the CUDA bias-gradient
  reduction path. The macro build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  slowed to `2833.33 ms` with steps `2823.81`, `2831.61`, and `2835.06 ms`.
  The SM120 bias-gradient block size remains `512`.
- Rejected `LLMK_SM120_BIAS_BLOCK_SIZE=448` for the CUDA bias-gradient
  reduction path. The macro build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  slowed to `2836.73 ms` with steps `2834.60`, `2840.38`, and `2833.09 ms`.
  The SM120 bias-gradient block size remains `512`.
- Rejected lowering the SM120 packed-QKV attention prep launch to
  `LLMK_SM120_DPREP_WARPS=1`. The macro build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  slowed to `2832.82 ms` with steps `2835.94`, `2823.95`, and `2841.69 ms`.
  The attention prep launch remains at `3` warps.
- Rejected lowering `LLMK_SM120_HUGE_N_THRESHOLD` to `2048` under the current
  K-tile 16 stack. The macro build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), and the focused benchmark improved
  a few attention/projection rows, but it badly regressed qkv forward
  (`1298.24 us`) and the MLP-up path. TinyStories 3-step validation slowed to
  `2825.23 ms` with steps `2810.98`, `2817.60`, and `2832.86 ms`, so the
  huge-N threshold remains `8192`.
- Rejected a temporary forward-only `LLMK_SM120_N96_K_TILE` hook for the
  128x96 SM120 forward tile. The `LLMK_SM120_N96_K_TILE=64` build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but the
  focused benchmark regressed the qkv/MLP N96 forward rows badly (qkv
  `1384.48 us`, fused FC `2023.97 us`) and TinyStories 3-step validation
  slowed to `2926.51 ms` with steps `2898.79`, `2928.46`, and `2924.56 ms`.
  The temporary hook was removed and the N96 forward tile again uses
  `LLMK_SM120_K_TILE=32`.
- Rejected a temporary forward-only `LLMK_SM120_N96_K_TILE=16` hook for the
  same 128x96 SM120 forward tile. The candidate built, but `test_matmul`
  failed before benchmark or TinyStories validation with an illegal memory
  access on the GPT-2 124M MLP-up forward row. The temporary hook was removed,
  so the N96 forward tile remains on the default `LLMK_SM120_K_TILE=32`.
- Rejected a scoped fused-GELU dispatch hook that disabled the SM120 N96 tile
  only for the MLP-up bias+GELU forward route while leaving qkv forward on N96.
  The `LLMK_SM120_FORWARD_GELU_N96=0` build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark
  regressed fused FC forward to `1947.07 us` versus cuBLASLt `1494.29 us`.
  TinyStories 3-step validation slowed to `2855.39 ms` with steps `2835.93`,
  `2856.66`, and `2854.13 ms`, so the temporary hook was removed.
- Rejected a temporary huge-N-forward-only swizzle hook. The
  `LLMK_SM120_HUGE_N_FORWARD_SUPER_M=10` build passed `test_matmul` (`8/8`)
  and `test_attention` (all three smoke shapes), but the focused benchmark did
  not improve LM-head forward (`27052.64 us` versus cuBLASLt `23490.75 us`)
  and worsened attention-projection dWeight to `556.20 us`. TinyStories
  3-step validation slowed to `2833.79 ms` with steps `2885.71`, `2818.30`,
  and `2849.28 ms`, so the forward-only hook was removed.
- Rejected retesting pure SM120 TK `LLMK_SM120_DWEIGHT_SUPER_M=6` on the
  current K-tile 16/dInput stack. The macro build passed `test_matmul` (`8/8`)
  and `test_attention` (all three smoke shapes), but the focused benchmark
  still trailed cuBLASLt on every dWeight row and worsened
  attention-projection dWeight to `493.91 us` versus cuBLASLt `327.87 us`.
  TinyStories 3-step validation averaged `2820.47 ms` with steps `2794.60`,
  `2805.93`, and `2835.00 ms`, so the source default remains
  `LLMK_SM120_DWEIGHT_SUPER_M=2`.
- Rejected `LLMK_SM120_BIAS_BLOCK_SIZE=480` for the CUDA bias-gradient
  reduction path. The macro build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but TinyStories 3-step validation
  slowed to `3015.93 ms` with steps `3031.65`, `3007.17`, and `3024.69 ms`.
  The SM120 bias-gradient block size remains `512`.
- Rejected `LLMK_SM120_DINP_SUPER_M=11` on the current K-tile 16/dWeight
  stack. The macro build compiled, but `test_matmul` failed the plain dInput
  row with max diff `5.2188` versus the `0.50` tolerance, so no focused
  benchmark or TinyStories validation was run. The dInput swizzle remains the
  source default `8`.
- Rejected increasing SM120 packed-QKV attention prep to
  `LLMK_SM120_DPREP_WARPS=5` on top of the current K-tile 16 stack. The macro
  build passed `test_matmul` (`8/8`) and `test_attention` (all three smoke
  shapes), but TinyStories 3-step validation slowed to `2831.42 ms` with steps
  `2863.52`, `2820.11`, and `2842.72 ms`. The attention prep launch remains
  at `3` warps.
- Rejected retesting pure SM120 TK `LLMK_SM120_DWEIGHT_SUPER_M=8` on the
  current K-tile 16/dInput stack. The macro build passed `test_matmul` (`8/8`)
  and `test_attention` (all three smoke shapes), but the focused benchmark
  still trailed cuBLASLt on every dWeight row and worsened fcproj dWeight to
  `1872.37 us`. TinyStories 3-step validation regressed sharply to
  `3491.02 ms` with steps `3503.58`, `3421.97`, and `3560.07 ms`, so the
  source default remains `LLMK_SM120_DWEIGHT_SUPER_M=2`.
- Rejected lowering `LLMK_SM120_HUGE_N_THRESHOLD` to `3072`. This avoided the
  qkv forward route hit by the rejected `2048` threshold and passed
  `test_matmul` (`8/8`) plus `test_attention` (all three smoke shapes), but
  the focused benchmark still trailed cuBLASLt on the material projection rows
  and worsened fcproj dWeight to `1673.40 us`. TinyStories 3-step validation
  regressed badly to `5340.61 ms` with steps `6528.14`, `5362.80`, and
  `5318.43 ms`, so the huge-N threshold remains `8192`.
- Rejected lowering the current qkv dWeight split-K route to
  `LLMK_SM120_DWEIGHT_SPLIT_K=8` on top of the K-tile 16/dInput stack. The
  macro build passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), but the focused benchmark worsened qkv dWeight to
  `1282.88 us` versus cuBLASLt `1031.05 us`, and TinyStories 3-step validation
  regressed to `4873.73 ms` with steps `4677.12`, `5353.01`, and `4394.45 ms`.
  The source default remains 16-way split-K for qkv and the wrapper's 8-way cap
  for non-QKV dWeight shapes.
- Rejected a temporary `LLMK_SM120_DWEIGHT_NON_QKV_SPLIT_K=4` hook that kept
  qkv dWeight at the default 16-way split while lowering non-QKV dWeight shapes
  from the wrapper's 8-way cap to 4-way split-K. The first `test_matmul` hit
  the known transient MLP-up row, the immediate rerun passed `8/8`, and
  `test_attention` passed all three smoke shapes, but the focused benchmark
  made attention-projection dWeight much worse (`718.82 us` versus cuBLASLt
  `327.40 us`). TinyStories 3-step validation regressed to `5437.90 ms` with
  steps `6876.11`, `5957.73`, and `4918.08 ms`, so the temporary hook was
  removed and non-QKV shapes remain capped at 8-way split-K.
- Rejected retesting pure SM120 TK `LLMK_SM120_DWEIGHT_SUPER_M=9` on the
  current K-tile 16/dInput stack. The first `test_matmul` hit the known
  transient MLP-up row, the immediate rerun passed `8/8`, and `test_attention`
  passed all three smoke shapes, but the focused benchmark still trailed
  cuBLASLt on every dWeight row, including qkv dWeight (`1219.95 us` versus
  `1025.24 us`) and attention-projection dWeight (`533.90 us` versus
  `348.77 us`). TinyStories 3-step validation regressed to `3048.72 ms` with
  steps `3051.72`, `3044.99`, and `3052.46 ms`, so the source default remains
  `LLMK_SM120_DWEIGHT_SUPER_M=2`.
- Rejected a temporary dWeight-specific N128 K-tile hook. The
  `LLMK_SM120_DWEIGHT_N128_K_TILE=32` build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark still
  trailed cuBLASLt on every dWeight row and worsened qkv dWeight to
  `1270.48 us` versus cuBLASLt `1055.14 us`. TinyStories 3-step validation
  regressed to `3055.98 ms` with steps `3045.03`, `3049.89`, and
  `3062.08 ms`, so the temporary hook was removed and the N128 dWeight route
  again uses `LLMK_SM120_HUGE_N_K_TILE=16`.
- Rejected an unguarded temporary `LLMK_SM120_DWEIGHT_N128_M256=1` hook that
  routed every N128 dWeight shape through the existing 256x128 / 8-warp tile.
  It passed `test_matmul` (`8/8`) and `test_attention` (all three smoke
  shapes), but `bench_sm120_matmul` aborted before timing because LM-head
  dWeight has `M=50304`, which is not divisible by the candidate's 256-row
  tile. No TinyStories validation was run; the hook was removed.
- Rejected a guarded version of the same 256x128 dWeight route that only used
  the 8-warp tile when `M % 256 == 0` and left LM-head on the existing 128x128
  route. The guarded hook passed `test_matmul` (`8/8`) and `test_attention`
  (all three smoke shapes), but it made the dWeight rows it touched materially
  worse, including qkv dWeight (`1475.85 us` versus cuBLASLt `1065.81 us`) and
  attention-projection dWeight (`599.74 us` versus `327.85 us`). TinyStories
  3-step validation regressed to `3154.36 ms` with steps `3205.86`, `3147.39`,
  and `3161.33 ms`, so the temporary hook was removed.
- Rejected `LLMK_SM120_GRAD_K_TILE=96` before runtime validation. The candidate
  would reduce dInput loop trips versus the default `64`, but ptxas rejected
  the 256x64 / 8-warp grad kernels for excessive shared memory (`0x1e000`
  bytes versus the SM120 limit `0x18c00`), so no smoke, benchmark, or
  TinyStories validation was run.
- Rejected `LLMK_SM120_GRAD_K_TILE=16`. The macro build completed, but
  `test_matmul` deterministically hit an illegal memory access on the fused
  dGELU dInput smoke row; an immediate rerun failed at the same row. No
  focused benchmark or TinyStories validation was run.
- Rejected disabling the SM120 fused forward-bias epilogue with
  `LLMK_SM120_FUSE_BIAS=0`. The macro build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark
  regressed the bias-bearing forward rows sharply, including qkv forward
  (`1543.98 us` versus cuBLASLt `1070.10 us`) and attention-projection forward
  (`513.88 us` versus `404.97 us`). TinyStories 3-step validation regressed to
  `3108.64 ms` with steps `3096.47`, `3109.00`, and `3108.28 ms`, so the
  fused bias epilogue remains enabled.
- Rejected disabling the SM120 fused MLP bias+GELU route with
  `LLMK_SM120_FUSE_GELU=0`. The first `test_matmul` run reported a transient
  GPT-2 MLP-up max diff of `7.5000`, but the immediate rerun passed `7/7`
  smoke rows and `test_attention` passed all three shapes. The focused
  benchmark still trailed cuBLASLt on the MLP, dInput, dWeight, and LM-head
  rows, and TinyStories 3-step validation regressed to `3090.25 ms` with steps
  `3081.57`, `3081.75`, and `3098.75 ms`, so fused MLP bias+GELU remains
  enabled.
- Rejected a temporary SM120 fused-GELU epilogue approximation hook. The
  `LLMK_SM120_FAST_GELU=1` build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark
  regressed fused FC forward to `1614.05 us` versus cuBLASLt `1498.27 us`, and
  TinyStories 3-step validation averaged `3043.15 ms` with steps `3047.18`,
  `3035.16`, and `3051.13 ms`. The hook also changed the early validation-loss
  path, so it was removed and the fused forward epilogue continues to use the
  exact `tanhf` expression.
- Rejected global forward swizzle `LLMK_SM120_SUPER_M=5`. The macro build
  passed `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes),
  and the focused benchmark improved only qkv forward (`1073.88 us` versus
  cuBLASLt `1101.44 us`) while still trailing the material dInput, dWeight,
  MLP, projection, and LM-head rows. TinyStories 3-step validation regressed to
  `3052.94 ms` with steps `3036.98`, `3040.83`, and `3065.06 ms`, so the
  global SM120 swizzle remains `9`.
- Rejected global forward swizzle `LLMK_SM120_SUPER_M=6`. The macro build
  passed `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes),
  but the focused benchmark still trailed cuBLASLt on the important dInput,
  dWeight, MLP, and LM-head rows despite a noisy attention-projection forward
  win. TinyStories 3-step validation averaged `2912.92 ms` with steps
  `2859.28`, `2923.24`, and `2902.60 ms`, still slower than the accepted
  source default and the llm.c baseline, so the global SM120 swizzle remains
  `9`.
- Rejected a temporary dWeight-only N96 route. The
  `LLMK_SM120_DWEIGHT_FORCE_N96=1` hook forced dWeight TN shapes with
  `N % 96 == 0` through the 128x96 tile while leaving dInput routing
  unchanged. It passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), but the focused benchmark worsened every dWeight row,
  including qkv dWeight (`1376.45 us` versus cuBLASLt `1028.39 us`) and
  LM-head dWeight (`28791.69 us` versus `21616.07 us`). TinyStories 3-step
  validation averaged `2936.82 ms` with steps `2942.24`, `2939.30`, and
  `2934.34 ms`, so the temporary hook was removed.
- Rejected a temporary dInput-only 8-warp 128x64 tile. The
  `LLMK_SM120_DINP_128X64_WARPS8=1` build completed, but `test_matmul`
  deterministically failed the plain dInput row with max diff `9.7812` on the
  first run and `11.1562` on the immediate rerun, versus the `0.50` tolerance.
  No focused benchmark or TinyStories validation was run.
- Rejected a temporary dInput-only 2-warp 128x64 tile. The
  `LLMK_SM120_DINP_128X64_WARPS2=1` build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark made
  every dInput row slower, including qkv dInput (`1383.49 us` versus cuBLASLt
  `1006.93 us`) and LM-head dInput (`28154.13 us` versus `21951.83 us`).
  TinyStories 3-step validation averaged `2874.72 ms` with steps `2866.70`,
  `2869.52`, and `2879.93 ms`, so the temporary hook was removed.
- Rejected a temporary 8-warp forward N96 tile. The
  `LLMK_SM120_FORWARD_N96_WARPS8=1` build completed, but `test_matmul`
  deterministically hit an illegal memory access on the GPT-2 124M MLP-up
  forward row; an immediate rerun failed at the same row. No focused benchmark
  or TinyStories validation was run.
- Rejected a temporary 2-warp forward N96 tile. The
  `LLMK_SM120_FORWARD_N96_WARPS2=1` build completed, but `test_matmul`
  deterministically failed the GPT-2 124M MLP-up forward row with max diff
  `7.1562` on the first run and `7.8750` on the immediate rerun, versus the
  `0.50` tolerance. No focused benchmark or TinyStories validation was run.
- Rejected an 8-warp version of the 128x128 dWeight N128 tile. The first
  `LLMK_SM120_DWEIGHT_N128_WARPS8=1` `test_matmul` run hit the recurring
  unrelated MLP-up forward row, the immediate rerun passed `8/8`, and
  `test_attention` passed all three smoke shapes. The focused benchmark
  regressed every dWeight row, including qkv dWeight (`1385.06 us` versus
  cuBLASLt `1032.96 us`) and LM-head dWeight (`29426.16 us` versus
  `22893.51 us`). TinyStories 3-step validation averaged `2906.92 ms` with
  steps `2911.05`, `2903.80`, and `2910.03 ms`, so the temporary hook was
  removed.
- Rejected a 2-warp version of the 128x128 dWeight N128 tile. The first
  `LLMK_SM120_DWEIGHT_N128_WARPS2=1` `test_matmul` run hit the recurring
  unrelated MLP-up forward row, the immediate rerun passed `8/8`, and
  `test_attention` passed all three smoke shapes. The focused benchmark made
  dWeight much slower across the board, including qkv dWeight (`2220.94 us`
  versus cuBLASLt `1042.81 us`) and LM-head dWeight (`43562.87 us` versus
  `21663.35 us`). TinyStories 3-step validation averaged `3403.15 ms` with
  steps `3389.46`, `3386.64`, and `3419.65 ms`, so the temporary hook was
  removed.
- Rejected a dWeight-specific N128 K-tile value of `8`. The temporary
  `LLMK_SM120_DWEIGHT_N128_K_TILE=8` hook failed at `test_matmul` compile time:
  ThunderKittens rejected the generated `st<bf16,8,128>` and matching register
  tiles because their row/column dimensions are not divisible by the required
  tile dimensions. No smoke, benchmark, or TinyStories 3-step validation was
  run for this invalid tile, and the N128 dWeight route again uses the shared
  `LLMK_SM120_HUGE_N_K_TILE=16` trait.
- Rejected a temporary small-M 64x128 dWeight N128 route. The
  `LLMK_SM120_DWEIGHT_N128_M64=1` build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark worsened
  the targeted attention-projection and fcproj dWeight rows (`524.02 us` and
  `1658.01 us`) versus the current 128x128 route. TinyStories 3-step validation
  averaged `2836.84 ms` with steps `2816.75`, `2827.87`, and `2845.82 ms`, so
  the temporary hook was removed.
- Rejected a forward-only SM120 swizzle hook. The temporary
  `LLMK_SM120_FORWARD_SUPER_M=8` route kept dInput and dWeight on the accepted
  swizzles and passed `test_matmul` (`8/8`) plus `test_attention` (all three
  smoke shapes). The focused benchmark improved qkv and fcproj forward rows,
  but attproj forward and the remaining dWeight rows still trailed cuBLASLt;
  the required TinyStories 3-step validation averaged `2799.31 ms` with steps
  `2779.75`, `2798.74`, and `2799.88 ms`, slower than the current pure-TK
  source default, so the hook was removed.
- Rejected a plain-dInput-only `K_TILE=96` route. The temporary
  `LLMK_SM120_DINP_K96=1` build passed the existing matmul and attention smokes,
  but the focused benchmark regressed every dInput row, including qkv
  (`1370.29 us` versus cuBLASLt `1048.70 us`) and LM-head (`29623.68 us`
  versus `21908.70 us`). TinyStories 3-step validation slowed to `3041.63 ms`
  with steps `3091.12`, `3020.36`, and `3062.89 ms`, so the hook was removed
  and plain dInput stays on the accepted `LLMK_SM120_GRAD_K_TILE=64` route.
- Rejected a small-M-only dWeight N128 swizzle split. The temporary
  `LLMK_SM120_DWEIGHT_SMALL_M_SUPER_M=3` route passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark did not
  improve the targeted small-M dWeight rows: attention-projection dWeight was
  `518.85 us` versus cuBLASLt `348.76 us`, and fcproj dWeight was
  `1572.26 us` versus `1419.39 us`. TinyStories 3-step validation averaged
  `2814.59 ms` with steps `2806.56`, `2803.64`, and `2825.53 ms`, so the hook
  was removed.
- Rejected pure SM120 TK codegen with `-Xptxas -dlcm=cg`. The build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but the
  focused benchmark stayed mixed with material dInput/dWeight rows behind
  cuBLASLt, and TinyStories 3-step validation regressed to `3552.81 ms` with
  steps `3526.77`, `3535.47`, and `3570.15 ms`. Pure SM120 TK builds keep the
  normal ptxas load-cache mode.
- Rejected pure SM120 TK codegen with `-Xptxas -maxrregcount=192`. The build
  passed `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes),
  but the focused benchmark worsened key dWeight rows such as qkv
  (`1278.28 us` versus cuBLASLt `1032.57 us`) and TinyStories 3-step validation
  regressed to `3052.12 ms` with steps `3156.48`, `3059.68`, and `3044.56 ms`.
  Pure SM120 TK builds keep the uncapped register allocation.
- Rejected a dWeight-specific N128 `K_TILE=64` route. The temporary
  `LLMK_SM120_DWEIGHT_N128_K_TILE=64` hook kept huge-N forward on K16 and
  passed `test_matmul` (`8/8`) plus `test_attention` (all three smoke shapes),
  but the focused benchmark made every dWeight row worse, including qkv
  `1410.43 us` versus cuBLASLt `1033.24 us` and LM-head `28096.62 us` versus
  `21501.94 us`. TinyStories 3-step validation regressed to `3163.98 ms` with
  steps `3171.29`, `3155.54`, and `3172.43 ms`, so the temporary hook was
  removed.
- Rejected a dWeight 128x64 K16 route. The temporary
  `LLMK_SM120_DWEIGHT_N64_K16=1` build passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), and the focused benchmark improved
  some direct dWeight rows such as attention-projection (`475.47 us`) and qkv
  (`1162.90 us`), but all dWeight rows still trailed cuBLASLt and TinyStories
  3-step validation regressed to `3077.65 ms` with steps `3013.40`, `3070.49`,
  and `3084.81 ms`. The temporary route was removed.
- Rejected narrowing that N64/K16 dWeight route to only the `M=768` rows. The
  `LLMK_SM120_DWEIGHT_M768_N64_K16=1` build passed the matmul and attention
  smokes and improved fcproj dWeight to `1499.45 us` in the focused benchmark,
  but the dWeight rows still trailed cuBLASLt and TinyStories 3-step validation
  regressed to `3007.03 ms` with steps `3020.74`, `3004.63`, and `3009.44 ms`.
  The temporary route was removed.
- Rejected narrowing the N64/K16 dWeight route to qkv only. The
  `LLMK_SM120_DWEIGHT_QKV_N64_K16=1` build passed the matmul and attention
  smokes and improved qkv dWeight to `1126.56 us` in the focused benchmark
  (`1.04x` slower than cuBLASLt), but TinyStories 3-step validation regressed
  badly to `3387.68 ms` with steps `3396.48`, `3372.17`, and `3403.20 ms`, so
  the temporary route was removed.
- Rejected pure SM120 TK codegen with `--extra-device-vectorization`. The
  build passed `test_matmul` (`8/8`) and `test_attention` (all three smoke
  shapes), but the focused benchmark still left every pure-TK GEMM row behind
  cuBLASLt and worsened key timings such as qkv dWeight (`1191.98 us` versus
  `1014.41 us`) and LM-head dWeight (`24070.96 us` versus `21794.65 us`).
  TinyStories 3-step validation regressed to `3013.36 ms` with steps
  `3031.73`, `3005.11`, and `3021.61 ms`, so pure SM120 TK builds keep the
  current default codegen flags.
- Rejected a temporary dWeight-only 96x128 TN tile route. The first
  `test_matmul` pass failed accumulated dWeight, but an immediate rerun passed
  `8/8` and `test_attention` passed all three smoke shapes. The focused
  benchmark then hit an illegal memory access during qkv dWeight timing after
  qkv forward/dInput, so no TinyStories 3-step validation was run and the
  temporary 96x128 hook was removed.
- Rejected a temporary dWeight-only 192x128 TN tile route. The candidate passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but
  `bench_sm120_matmul` again hit an illegal memory access during qkv dWeight
  timing before any useful dWeight measurement. No TinyStories 3-step
  validation was run, and the temporary 192x128 hook was removed.
- Rejected an isolated dWeight-only N-swizzle grid traversal with
  `LLMK_SM120_DWEIGHT_SUPER_N=6`. The corrected TN-only hook passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but
  the focused benchmark worsened material dWeight rows (qkv `1318.50 us`,
  attention projection `517.49 us`, MLP-up `1703.62 us`) and TinyStories
  3-step validation regressed to `3391.70 ms` with steps `3499.31`,
  `3391.40`, and `3392.00 ms`, so the hook was removed.
- Rejected a qkv-forward-only swizzle route with
  `LLMK_SM120_QKV_FORWARD_SUPER_M=8`. The candidate passed `test_matmul`
  (`8/8`) and `test_attention` (all three smoke shapes), but the focused
  benchmark only tied qkv forward (`1147.98 us` versus cuBLASLt `1143.10 us`)
  while leaving dInput and dWeight behind. TinyStories 3-step validation
  regressed to `3017.52 ms` with steps `3015.67`, `3004.43`, and `3030.61 ms`,
  so the wrapper and benchmark hooks were removed.
- Rejected a cuBLASLt-backed SM120 workspace retest with
  `LLMK_SM120_CUBLASLT_WORKSPACE_MB=64`. The build passed `test_matmul`
  (`8/8`) and `test_attention` (all three smoke shapes), but the focused
  benchmark did not improve the remaining pure-TK gaps and TinyStories 3-step
  validation averaged `2568.10 ms` with steps `2563.89`, `2561.70`, and
  `2574.50 ms`, slower than the accepted no-master cuBLASLt default.
- Rejected the combined pure-TK dWeight macro
  `LLMK_SM120_DWEIGHT_SPLIT_K=32` with `LLMK_SM120_DWEIGHT_SUPER_M=1`. It
  passed `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes)
  and improved qkv dWeight to `1160.91 us`, but it worsened smaller dWeight
  rows and still trailed cuBLASLt. TinyStories 3-step validation regressed to
  `3017.20 ms` with steps `3016.19`, `3013.50`, and `3020.89 ms`, so the
  source defaults remain split-K `16` and dWeight `SUPER_M=2`.
- Rejected `LLMK_SM120_BIAS_BLOCK_SIZE=128` for the CUDA bias-gradient
  reduction path. The macro build failed `test_matmul` twice on the GPT-2 MLP
  up-projection forward row (`7.5078` then `6.8633` max diff versus `0.50`
  tolerance), so no focused benchmark or TinyStories 3-step validation was run.
  The SM120 bias-gradient block size remains `512`.
- Rejected `LLMK_SM120_DINP_SUPER_M=3` under the current K-tile 16/dWeight
  stack. The macro build passed `test_matmul` (`8/8`) and `test_attention`
  (all three smoke shapes), but the focused benchmark did not improve the
  material dInput rows and TinyStories 3-step validation regressed to
  `3358.86 ms` with steps `3349.12`, `3366.50`, and `3351.22 ms`. The dInput
  swizzle remains `8`.
- Rejected `LLMK_SM120_DWEIGHT_SUPER_M=11`. The first `test_matmul` pass hit
  the known transient MLP-up forward row, the immediate rerun passed `8/8`,
  and `test_attention` passed all three smoke shapes. The focused benchmark
  still trailed cuBLASLt on every dWeight row, and TinyStories 3-step
  validation regressed to `2982.84 ms` with steps `3033.89`, `3038.07`, and
  `2927.62 ms`. The dWeight swizzle remains `2`.
- Rejected `LLMK_SM120_DWEIGHT_SUPER_M=13`. The macro build passed
  `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes), but
  the focused benchmark still trailed cuBLASLt on every dWeight row and
  TinyStories 3-step validation regressed to `3030.43 ms` with steps
  `3019.57`, `3026.12`, and `3034.73 ms`. The dWeight swizzle remains `2`.
- Rejected `LLMK_SM120_DWEIGHT_SUPER_M=15`. The first `test_matmul` pass
  failed the known transient MLP-up row and the dWeight row, the immediate
  rerun passed `8/8`, and `test_attention` passed all three smoke shapes. The
  focused benchmark worsened the dWeight rows versus the accepted `2` default,
  and TinyStories 3-step validation regressed to `3026.30 ms` with steps
  `3018.87`, `3027.63`, and `3024.97 ms`. The dWeight swizzle remains `2`.
- Rebaselined the clean no-extra-macro pure SM120 TK source default after the
  rejection-only rounds. `test_matmul` passed `8/8`, `test_attention` passed
  all three smoke shapes, and the focused benchmark still showed pure TK behind
  cuBLASLt on the material dInput/dWeight and LM-head rows. TinyStories
  3-step validation completed with the expected finite loss/norm trace but
  averaged `3384.57 ms` with steps `3379.52`, `3382.37`, and `3386.77 ms`,
  so the pure-TK kernel-outperformance goal remains open.
- Rejected a global direct column-layout shared-to-register load hook for the
  SM120 NN and TN GEMM kernels. The build completed, but `test_matmul` failed
  the fused dGELU dInput row with max diff `2.4453` versus the `0.50`
  tolerance, so no focused benchmark or TinyStories 3-step validation was run.
  The temporary hook was removed.
- Rejected a scoped direct column-layout shared-to-register load retry that
  kept fused dGELU on the existing register-swap path while direct-loading
  plain dInput and dWeight tiles. It passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), and the focused benchmark improved
  qkv forward/dInput to `0.98x`/`0.96x` of cuBLASLt, but every material
  dWeight row and LM-head forward still trailed cuBLASLt. TinyStories
  3-step validation averaged `2889.63 ms` with steps `2884.80`, `2882.86`,
  and `2896.39 ms`, slower than the supplied llm.c baseline, so the temporary
  hook was removed.
- Rejected isolating the direct column-layout load to plain NN dInput only.
  The candidate passed `test_matmul` (`8/8`) and `test_attention` (all three
  smoke shapes), and the focused benchmark made qkv and attention-projection
  dInput faster than cuBLASLt, but fc, fcproj, and LM-head dInput still trailed
  and the material dWeight rows were still behind. TinyStories 3-step
  validation regressed to `3585.88 ms` with steps `3569.84`, `3594.94`, and
  `3576.82 ms`, so the temporary hooks were removed.
- Rejected isolating the direct column-layout load to TN dWeight only. It
  passed `test_matmul` (`8/8`) and `test_attention` (all three smoke shapes),
  but the focused benchmark still left every dWeight row behind cuBLASLt and
  worsened several dInput rows. TinyStories 3-step validation averaged
  `3205.80 ms` with steps `3209.82`, `3208.68`, and `3202.92 ms`, and shifted
  the norm trace, so the temporary hook was removed.
- Rejected isolating the stricter `cp.async` wait policy to the SM120 NN
  dInput kernel. The wait-1 candidate passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark
  worsened every dInput row and TinyStories 3-step validation regressed to
  `4044.97 ms` with steps `4047.73`, `4026.52`, and `4063.41 ms`. The
  temporary hook was removed.
- Rejected isolating the stricter `cp.async` wait policy to the SM120 NT
  forward kernel. The wait-1 candidate passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark
  regressed the forward rows, including qkv (`1316.94 us`) and LM-head
  (`33061.30 us`). TinyStories 3-step validation averaged `3675.21 ms` with
  steps `3717.45`, `3699.34`, and `3651.08 ms`, so the temporary hook was
  removed.
- Rejected direct column-layout loading for the SM120 attention forward V tile.
  The candidate passed `test_attention` and passed `test_matmul` on rerun after
  the recurring unrelated MLP-up transient, but TinyStories 3-step validation
  regressed catastrophically to `17988.24 ms` with steps `17063.37`,
  `17795.62`, and `18180.86 ms`. The temporary hook was removed.
- Rejected a small-M-only dWeight N128 swizzle retest with
  `LLMK_SM120_DWEIGHT_SMALL_M_SUPER_M=6`. The candidate eventually passed
  `test_matmul` after two unrelated transient smoke failures and passed
  `test_attention`, but the focused benchmark did not close the targeted
  attention-projection or fcproj dWeight gaps. TinyStories 3-step validation
  regressed catastrophically to `16645.77 ms` with steps `16974.58`,
  `16367.83`, and `16923.70 ms`, so the temporary hook was removed.
- Rejected macro-only `LLMK_SM120_DWEIGHT_SUPER_M=18`, which groups the full
  qkv dWeight M-tile span. It passed `test_matmul` (`8/8`) and
  `test_attention` (all three smoke shapes), but the focused benchmark still
  left all dWeight rows behind cuBLASLt and worsened fc/fcproj dWeight.
  TinyStories 3-step validation regressed to `4642.61 ms` with steps
  `10255.57`, `4636.76`, and `4648.47 ms`, so the source default remains `2`.

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
