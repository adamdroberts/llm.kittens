# Changelog

Append-only history of meaningful changes to llm.kittens. Roughly grouped by
milestone. Adds within a milestone are listed in chronological order.

The canonical "what is done / what is left" is [`goal.md`](goal.md). The
changelog is the diary; `goal.md` is the plan.

## 2026-05 — M8 profiling gate hardening

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
- Tightened `scripts/validate_goal_h100.sh preflight` so runtime gates require
  H100/sm90-class GPUs. Unsupported devices such as RTX 5090 fail before
  NCCL/MPI checks unless `ALLOW_NON_H100=1` is set for dry compile/debug runs.
- Matched the standalone `cuda-runtime` probe to the same H100/sm90-class
  contract, so running that phase directly cannot accept sm_120 or other
  unsupported GPUs as runtime evidence.
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
