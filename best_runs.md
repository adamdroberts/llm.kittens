# Best Runs - SM120 Kernel Benchmarks

This is the live scoreboard for SM120 / RTX 5090 GPT-2 kernel selection. A row
is promotable only when the relevant correctness gate passes, the focused
benchmark beats the current provider for that shape, and the TinyStories smoke
does not regress after trainer integration.

## Benchmark Targets

Run a full SM120 optimization round on the RTX 5090 target with:

```bash
scripts/run_sm120_optimization_round.sh
```

Useful overrides:

```bash
RUN_LABEL=attn_candidate MAX_STEPS=3 scripts/run_sm120_optimization_round.sh
RUN_TRAINING=0 scripts/run_sm120_optimization_round.sh
RUN_STACK_PROBE=0 scripts/run_sm120_optimization_round.sh
KEEP_CHECKPOINTS=1 MAX_STEPS=10 scripts/run_sm120_optimization_round.sh
python3 dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/<run-label> \
    --require-manifest --require-stack-probe --require-correctness \
    --require-benchmarks --require-training \
    --write-scoreboard scratch/sm120_rounds/<run-label>/scoreboard-candidates.md
```

The harness writes build/test/benchmark/training logs under
`scratch/sm120_rounds/<run-label>/`, writes trainer output to
`log124M/5090_S_<run-label>` by default, and removes bulky `model_*.bin` /
`state_*.bin` checkpoint outputs from that round directory unless
`KEEP_CHECKPOINTS=1` is set. Non-dry runs also validate the captured artifacts
with `dev/validate_sm120_round.py`, write parsed candidate rows to
`scratch/sm120_rounds/<run-label>/scoreboard-candidates.md`, write the
machine-readable selected backend rows to
`scratch/sm120_rounds/<run-label>/selected-backends.json`, write the
non-trainer-callable selected winners to
`scratch/sm120_rounds/<run-label>/promotion-candidates.json`, and record
optional backend stack availability in `backend-stacks.json` / `backend-stacks.md`.
Each non-dry run also writes `round-manifest.json` / `round-manifest.md` with
the run config, commit, changed-path count, toolchain probe output, and SHA256
identity for the built smoke/benchmark/trainer binaries. Scoreboard rows should
cite that manifest plus the relevant correctness, benchmark, and training logs.
The validator requires every expected smoke-test, benchmark, and trainer binary
to exist in that manifest with SHA256 evidence, so a round cannot pass with a
missing executable identity.
The shared contract lists for allowed stacks, required families, runtime
kernels, correctness targets, and expected manifest binaries live in
`dev/sm120_objective_contract.py`; the probe, manifest writer, and validator
all import it. The same contract also owns the default native and optional-stack
round paths used by `dev/write_sm120_current_selection.py` and
`dev/audit_sm120_optimization_goal.py`, so the current-selection replay path
has one source of truth for the audited x10 native evidence and optional-stack
comparison round.
The current optional-stack comparison round is
`scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521`.
It includes the full correctness smoke set plus Torch, LibTorch C++, Triton,
cuDNN, and CuTeDSL comparison logs. The validator reports `203` benchmark rows,
`43/43` Torch objective rows, `17/17` Torch runtime/classifier rows, `5/5`
LibTorch C++ runtime timing rows, and `5/5` LibTorch C++ runtime parity rows.
The current audit also requires supplemental LibTorch C++ GELU evidence from
`scratch/sm120_rounds/libtorch_runtime_gelu_20260521/bench_sm120_libtorch_runtime.log`,
with `1/1` supplemental timing and parity rows for `gelu_forward BT=65536 C=3072`.
It also requires the standalone trainer-link probe at
`scratch/sm120_rounds/libtorch_trainer_link_20260521/validate_libtorch_trainer_link.log`,
which proves a LibTorch executable can link and run the cached-`from_blob`
zero/copy route without `torch_python`.
The combined selection compares each optional-stack row against the current
native trainer-backed row before reporting a project-wide fastest situation.
In the current artifact, Torch-family rows win `9` project-wide exact
situations: separated-Q/K/V attention forward/backward, Torch-native
partial/non-trainer LayerNorm rows, non-equivalent AdamW reference rows, and
Torch C++ runtime memory rows. The older Python Torch GELU-forward row is no
longer listed as project-fastest because the current native CUDA trainer row is
faster. No Torch row is promoted into the trainer by default yet because each
faster row is either a layout rewrite, operator/reference prototype, contract
variant, or a sub-percent LibTorch C++ memory edge whose explicit trainer route
has not beaten the native x10 stability baseline.
When `--require-stack-probe` is set, the validator fails unless the probe
records every allowed stack from `optimise-goal.md` with evidence and a next
action: ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, Triton, Torch, CuTeDSL,
and Plain CUDA, plus GPU runtime evidence. The probe also records a per-family
applicability matrix for every required objective family and allowed stack so
not-applicable stacks are documented with a reason instead of silently skipped.
When `--require-benchmarks` is set, the validator also emits an objective
coverage table and fails if any required family from `optimise-goal.md` is
missing parsed timing rows. It also emits a GEMM shape coverage table and
fails if the matmul benchmark omits a required GPT-2 row: qkv, attention
projection, MLP up, MLP projection, or LM-head across the relevant forward,
dInput, dWeight, and accumulated dWeight passes. It also emits a GEMM provider
coverage table and fails if the matmul benchmark omits ThunderKittens,
cuBLASLt, or cuBLAS timing rows for any required GEMM pass and GPT-2 shape.
The validator also emits runtime shape coverage for required runtime-family
rows, including standalone bias-add widths, GPT-2 bias-gradient widths
`OC=768`, `OC=2304`, and `OC=3072`, plus hidden-state and logits-sized
`cudaMemsetAsync` / device-to-device copy rows, and the trainer gradients-zero
`cudaMemsetAsync` shape `grad_elems=124475904`.
When the stack matrix is present, it also emits baseline-provider coverage and
fails if a required family is benchmarked under a stack other than its recorded
baseline provider.
For Python-stack rounds, the validator now also emits `Torch Objective
Benchmark Coverage` and fails unless Torch has timing rows for all `43` exact
GPT-2 objective rows from the shared contract. This is the project-wide guard
that Torch was benchmarked against the native and optional kernel stacks before
any "Torch wins here" row is selected or rejected.
The validator also emits a `Selected Backend Rows` table and matching
`selected-backends.json` from the parsed benchmark logs. These artifacts pick
the fastest observed stack for each exact suite/kernel/shape row and record its
use scope plus a `trainer_call_path_kind`. Torch rows are selected where they
actually win, but rows that are only Python/operator prototypes, LibTorch
raw-pointer prototypes, or profiler-runtime copy evidence are marked as such
instead of silently implying a trainer promotion. Selected LibTorch C++ memory rows cite the same
`bench_sm120_libtorch_runtime.log` as timing and parity evidence, because that
log verifies full-row zero/copy and gradients-zero behavior before reporting timings. That probe
now defaults to cached `from_blob` wrappers over existing CUDA pointers, so its
LibTorch C++ rows exercise the raw-pointer route shape a linked trainer would
need rather than only ordinary Python-owned tensors. It also has a
`--route cxx-api-raw-pointer` mode that builds a standalone LibTorch C++ API
shared library without `torch_python` or pybind in the timed path, so the next
memory-route refresh can test the linked-trainer dependency shape directly.
The same runtime probe now measures GELU-forward through LibTorch C++ as a
supplemental runtime row; the audit verifies that evidence separately from the
older optional round's memory-row contract.
Future rounds also record `run_libtorch_trainer_link_probe`; when it is set,
the validator requires a standalone LibTorch executable probe log with compile
and runtime PASS markers before a LibTorch memory route can be treated as
trainer-link evidence.
Current
selection audit also cross-checks each selected row's source run label,
artifact directory, git commit, and copied `source_run_config` against the
referenced `round-manifest.json`, so mixed-stack rows cannot drift away from
their source run evidence. It also checks each source round's
`selected-backends.json` rows against the generated `Selected Backend Rows`
scoreboard table, checks the source selected-backend schema, run identity,
selection policy, benchmark counts, and Torch/LibTorch coverage metadata,
checks that the project-wide fastest and Torch-fastest rows preserve the stack,
timing, scope, and provenance fields from the optional round's
`selected-backends.json`, and checks that native trainer rows
preserve the source native selected row or the rejected-winner plus fallback
mapping for inactive native decisions. It also fails if any `cuda_copy_d2d`
runtime row is not marked as `profiler_runtime_benchmark_only`. Native and optional attention route
totals are checked against the source `attention_route_rows` in each round's
`selected-backends.json`. Resolved optional decisions are also checked against
their source `promotion-candidates.json` row, falling back to the optional
selected-backend row for trainer-callable inactive decisions. The current
selection also emits a `Fastest Rows Not Used By Trainer` debt table, and the
audit checks its call-path and decision-status counts so `active_promotions=0`
cannot be mistaken for complete optimization. Trainer/C++ callable rows in that
debt table must also carry trainer-smoke, x10, or stability evidence before
being resolved away from the trainer. The current selection also emits a
Torch-fastest disposition table; every Torch row that wins a project-wide exact
situation must be labeled as trainer-used, resolved away, or extra benchmark
evidence and must carry the action or reason for that disposition. The same pass emits a `Promotion Backlog` table and
`promotion-candidates.json`, grouped so native/direct and codegen integrations
come before library integrations, layout rewrites, and reference/state gaps,
then sorted by measured edge versus the next observed stack when available.
Torch/Triton wins without a trainer call path become the next integration
backlog rather than buried scoreboard rows, but separated-Q/K/V reference wins
and libtorch-sized dependency work no longer crowd out rows that fit the current
trainer structure more directly.
Resolved rows live in `dev/sm120_promotion_decisions.json`. The validator loads
that registry by default, keeps the original evidence in `promotion_candidates`,
and writes only still-actionable rows to `active_promotion_candidates` and the
active `Promotion Backlog`. This prevents a refreshed/rejected Torch, Triton, or
layout-only row from repeatedly reappearing as the next implementation target.
If a round has selected non-trainer-callable rows but all of them are covered by
the decision registry, the scoreboard states that no active promotion candidates
remain and still lists the resolved decisions below it. That is the signal to
move on to new stack probes or native candidates rather than re-integrating old
Torch/Triton rows.

Build just the repeatable local benchmark set with:

```bash
make -j bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm \
    bench_sm120_runtime \
    DEVICE_ARCH=SM120 \
    NO_MULTI_GPU=1 \
    NO_USE_MPI=1
```

`bench_sm120_matmul`, `bench_sm120_attention`, `bench_sm120_layernorm`, and
`bench_sm120_runtime` report the median of repeated event samples by default. Set
`LLMK_BENCH_REPEATS=<n>` to increase or reduce that repeat count for focused
A/B work.
The SM120 round harness also runs Python stack benchmarks when
`RUN_PYTHON_STACK_BENCHMARKS=1` (the default follows `RUN_BENCHMARKS`):
`dev/bench_sm120_torch_matmul.py` for Torch GEMM rows,
`dev/bench_sm120_cutedsl_matmul.py` for the CuTeDSL BF16 GEMM feasibility row,
`dev/triton/bench_sm120_matmul.py` for Triton GEMM and fused-epilogue rows,
`dev/bench_sm120_torch_attention.py` for native, packed, and materialized-packed Torch SDPA rows,
`dev/bench_sm120_cudnn_attention.py` for cuDNN SDPA feasibility rows,
`dev/triton/bench_sm120_attention.py` for Triton attention forward rows,
`dev/bench_sm120_torch_classifier.py` for Torch classifier rows,
`dev/triton/bench_sm120_classifier.py` for Triton classifier loss and dlogits rows,
`dev/triton/bench_sm120_layernorm.py` for Triton/Torch LayerNorm rows,
`dev/triton/bench_sm120_runtime.py` for Triton pointwise runtime rows, and
`dev/bench_sm120_torch_runtime.py` for Torch runtime-family rows, and
`dev/bench_sm120_libtorch_runtime.py` for LibTorch C++ API zero/copy feasibility
rows. `LLMK_LIBTORCH_RUNTIME_ROUTE` controls that LibTorch run; the harness
defaults to `cxx-api-raw-pointer` so future optional-stack rounds exercise the
standalone C++ API route, while `raw-pointer` can replay the older
extension-backed route.

The current combined SM120 selection artifact is regenerated with:

```bash
python3 dev/write_sm120_current_selection.py \
    --json-out scratch/sm120_rounds/current-sm120-selection.json \
    --markdown-out scratch/sm120_rounds/current-sm120-selection.md
```

It merges the stable native x10 selection with the latest optional-stack
comparison round, applies inactive decision-registry rows to the effective
trainer mix, and fails if an optional-stack promotion candidate is still active.
The native round must include TinyStories training evidence in its manifest and
`train_gpt2cu.log`; benchmark-only native rounds are accepted only with
`--allow-benchmark-only-native` for inspection artifacts, not for the published
current trainer mix.
It also fails if any selected optional-stack row without a trainer call path
lacks a matching promotion-candidate row or inactive resolved decision.
The Markdown artifact lists each resolved optional-stack decision so Torch,
Triton, and other optional wins can be reviewed by exact suite/kernel/shape,
scope, and decision status.
`--self-test` covers those contracts with synthetic pass/fail artifacts.

Audit the project-wide SM120 optimization evidence with:

```bash
python3 dev/audit_sm120_optimization_goal.py \
    --json-out scratch/sm120_rounds/current-sm120-audit.json \
    --markdown-out scratch/sm120_rounds/current-sm120-audit.md
```

The audit enforces that Torch is present in the objective stack matrix, that
the current selection records native TinyStories training evidence, that all
required Python-stack benchmark logs from the shared objective contract are
present and non-empty, that every selected optional-stack row without a trainer
call path has a matching promotion-candidate row and inactive resolved decision,
that the optional-stack scoreboard records `43/43` exact Torch objective
benchmark rows, that the LibTorch C++ runtime-memory log records timing and
parity evidence for all exact hidden/logits zero/copy and gradients-zero rows, that runtime copy
rows carry the profiler-only call-path kind, and that no active promotion candidate remains
before the current trainer mix is reported. It also checks that the current-selection artifact's optional
non-trainer row counts match the audited optional round, that the resolved
optional-decision table is present, and that the fastest-row debt summary
matches the actual resolved-away rows. The same audit replays the stack-probe
schema for both source rounds: every objective stack plus GPU runtime needs
status, evidence, candidate-use, and next-action text, and every family/stack
row needs status, reason, and next action. It also checks each source round's
manifest for SM120 config, git/toolchain metadata, expected smoke/benchmark/
trainer binaries, SHA256 evidence, and the optional Python-stack benchmark flag.
Set `RUN_CURRENT_SELECTION_AUDIT=1` on `scripts/run_sm120_optimization_round.sh`
to regenerate the current selection and audit artifacts through the same round
harness. By default the harness does not pass `--native-round` or
`--optional-round`, so the Python tools use the shared contract defaults. Set
`SM120_SELECTION_NATIVE_ROUND` and `SM120_SELECTION_OPTIONAL_ROUND` only when
auditing a newly generated artifact pair. `DRY_RUN=1` still prints the
selection/audit commands, which is the quick check that the harness is using the
contract defaults rather than stale copied paths.

Correctness gates before timing promoted candidates:

```bash
make -j test_matmul test_attention test_layernorm \
    test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm \
    train_gpt2cu \
    DEVICE_ARCH=SM120 \
    NO_MULTI_GPU=1 \
    NO_USE_MPI=1
```



## Recent Decisions


| Date       | Area                         | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Evidence                                                                                                                                                                                                                                                                                                                                               |
| ---------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2026-05-22 | TK approximate dGELU trainer route | Rejected `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` plus `LLMK_SM120_APPROX_DGELU_TANH=1` before training. This closes the remaining approximate-dGELU variant of the trainer-callable TK fcproj dInput+dGELU route; it cannot be promoted because the correctness gate fails. | `scratch/sm120_rounds/codex_sm120_tk_dgelu_approx_tanh_x3_20260522/test_matmul.log` passed 9/10 rows but failed the GPT-2 fcproj fused dGELU dInput row with `max abs diff = 0.5000` at tolerance `0.50`. Candidate `train_gpt2cu` hash was `549736f9...`; selected stack restored to `4374a593...` and all nine focused smokes passed. |
| 2026-05-22 | Post-TK-dGELU restore direct control | Reran `train-sm120.sh` after restoring from the rejected TK approximate dGELU build. The selected backend mix was active, but timing stayed in the recent slower direct-script band. | `scratch/sm120_rounds/direct_train_sm120_4374_post_tk_dgelu_restore_x10_20260522/train-sm120-summary.md` records active binary `4374a593...`, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar grad scale, and `zero_stage = 1`. x10 trainer avg was `2498.545 ms`, visible first3 `2487.257 ms`, and first5 `2489.754 ms`; generated step-10 checkpoints are retained in `log124M/5090_S`. |
| 2026-05-22 | Restored selected-source direct rerun retry | Reran `train-sm120.sh` again after the user reported their own faster retry and suspected prior GPU contention. This is the best recent Codex rerun for the restored `e2c85370...` binary, but it still did not reproduce the user's `~2462 ms` first-five band. | `scratch/sm120_rounds/direct_train_sm120_e2c_user_busy_rerun2_x10_20260522/train-sm120-summary.md` records CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar grad scale, fused GELU, and `zero_stage = 1`. x10 trainer avg was `2496.018 ms`, visible first3 `2486.363 ms`, and first5 `2488.054 ms`; generated step-10 checkpoints are retained in `log124M/5090_S`. |
| 2026-05-22 | Restored selected-source direct rerun | Reran `train-sm120.sh` after restoring from the global-norm candidate. The selected-source rebuild changed the binary hash to `e2c85370...` but preserved the intended backend startup mix; performance stayed in the same direct-script band and did not reproduce the user's faster observation. | `scratch/sm120_rounds/direct_train_sm120_e2c_restored_rerun_x10_20260522/train-sm120-summary.md` records CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar grad scale, and `zero_stage = 1`. x10 trainer avg was `2498.285 ms`, visible first3 `2489.113 ms`, and first5 `2491.232 ms`; generated step-10 checkpoints are retained in `log124M/5090_S`. |
| 2026-05-22 | Global-norm block768 sweep | Rejected `LLMK_SM120_GLOBAL_NORM_BLOCK_SIZE=768` on the selected stack. The focused global-norm runtime row was normal, but end-to-end training was slower than the selected direct-script rerun and `new-goal.md`, so the default 512-thread global-norm block remains selected. | `scratch/sm120_rounds/codex_sm120_global_norm_block768_x3_20260522` passed all nine focused smokes and validated. TinyStories x3 averaged `2495.645 ms`, visible first3 `2496.820 ms`; startup confirmed CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar grad scale, and `zero_stage = 1`. Selected binary was rebuilt without the candidate flag to hash `e2c85370...`; all nine focused smokes passed and a one-step startup probe confirmed the selected backend mix. |
| 2026-05-22 | Matmul dbias block-size sweep | Rejected `LLMK_SM120_BIAS_BLOCK_SIZE=768` on the selected stack. This closes the gap between default 512 and the already-rejected 256/1024 dbias reduction block sizes; the old upstream-style 768-thread choice is correctness-clean but slower end-to-end. | `scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522` passed all nine focused smokes and validated. TinyStories x3 averaged `2497.894 ms`, visible first3 `2498.463 ms`; startup confirmed CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar grad scale, and `zero_stage = 1`. Selected binary restored to `a60e97a6...`; all nine focused smokes passed. |
| 2026-05-22 | Exact selected-stack rerun after user fast band | Reran the exact `./train-sm120.sh` path again after restoring the selected binary. Startup confirmed the intended CUDA/Torch mix, but this process did not reproduce the user's `2461 ms` first-three band, so the faster user observation remains promising but not reproduced evidence. | `scratch/sm120_rounds/direct_train_sm120_codex_rerun_after_user_fast_20260522/train-sm120.log` used `train_gpt2cu` sha256 `a60e97a6...`, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar grad scale, maxconn1, and `-z 1`. Trainer avg was `2498.704 ms`; visible first3 `2489.237 ms`; first5 `2491.620 ms`, which is `19.394 ms` slower than `new-goal.md` first3 and `27.787 ms` slower than the user-pasted first3. Generated step-10 checkpoints were removed. |
| 2026-05-22 | No-ZeRO plus profiler-disable composition | Rejected composing single-GPU `-z 0` with `LLMK_DISABLE_CUDA_PROFILER`. This was a real trainer run of the component interaction, not a scorecard-only choice, and it regressed versus the selected stack and both component rows. | `scratch/sm120_rounds/codex_sm120_zero0_disable_profiler_x3_20260522` passed all nine focused smokes and validated. TinyStories x3 averaged `2496.472 ms`, visible first3 `2496.810 ms`; startup confirmed CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar grad scale, and `zero_stage = 0`. Selected binary restored to `a60e97a6...`; all nine focused smokes passed. |
| 2026-05-22 | Precompute grad-scale plus profiler-disable composition | Rejected composing `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` with `LLMK_DISABLE_CUDA_PROFILER`. Correctness and benchmarks passed, but the trainer result was slower than selected and slower than the independent near-current rows. | `scratch/sm120_rounds/codex_sm120_precompute_disable_profiler_x3_20260522` passed all nine focused smokes and validated. TinyStories x3 averaged `2496.255 ms`, visible first3 `2496.837 ms`; startup confirmed CUDA-kernel grad-zero, Torch C++ dresidual-zero, precomputed device AdamW scalar, and `zero_stage = 1`. Selected binary restored to `a60e97a6...`; all nine focused smokes passed. |
| 2026-05-22 | User-suggested selected-stack rerun | Reran the exact selected `train-sm120.sh` path again after the user noted the prior run may have hit a busy-GPU window. The GPU was idle before launch and the trainer printed the selected backend mix, but this still did not reproduce the user's pasted 2461 ms first-three band or beat `new-goal.md`. | `scratch/sm120_rounds/direct_train_sm120_aa2_user_rerun_verify10_x10_20260522/train-sm120.log` used `train_gpt2cu` sha256 `aa2d2499...`, CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar grad scale, maxconn1, and `-z 1`. x10 averaged `2490.359 ms`; first3 averaged `2481.190 ms`, which is `11.347 ms` slower than `new-goal.md` and `19.740 ms` slower than the user-pasted first3. Generated step-10 checkpoints were removed. |
| 2026-05-22 | Attention bwd64 tile sweep | Rejected `LLMK_SM120_ATTN_BWD_BLOCK=64` on the selected trainer stack. This closes the high-side attention backward tile gap in the current scorecards; correctness passes, but the backward benchmark and trainer timing regress badly. | `scratch/sm120_rounds/codex_sm120_attn_bwd64_recovered_x3_20260522` passed all nine focused smokes, then measured attention `787.890/18050.102 us` and TinyStories steps `3932.81, 3901.03, 3902.40 ms` with visible first3 `3912.080 ms`. That is `1430.890 ms` slower than the latest selected direct rerun and `1442.237 ms` slower than `new-goal.md`. Restored selected binary hash `407bd0a4...` passed all nine focused smokes. |
| 2026-05-22 | Attention fwd64 tile sweep | Rejected `LLMK_SM120_ATTN_FWD_BLOCK=64` on the selected trainer stack. This closes the high-side attention forward tile gap in the current scorecards; correctness passes, but the forward benchmark and trainer timing regress badly. | `scratch/sm120_rounds/codex_sm120_attn_fwd64_recovered_x3_20260522` passed all nine focused smokes, then measured attention `3223.705/2742.139 us` and TinyStories steps `2717.10, 2711.08, 2713.86 ms` with visible first3 `2714.013 ms`. That is `234.627 ms` slower than the latest selected direct rerun and `244.170 ms` slower than `new-goal.md`. Restored selected binary hash `aa2d2499...` passed all nine focused smokes. |
| 2026-05-22 | Dprep lower-bound sweep | Rejected `LLMK_SM120_DPREP_WARPS=1`. The forward attention microbenchmark improved versus dprep=2, but backward attention and trainer timing regressed materially, closing the low-side dprep sweep around the selected dprep=3 setting. | `scratch/sm120_rounds/codex_sm120_dprep1_recovered_x3_20260522` passed all nine focused smokes and measured attention `784.284/2857.293 us`. TinyStories steps were `2502.26, 2493.90, 2496.10 ms`, visible first3 `2497.420 ms`, which is `18.033 ms` slower than the latest selected direct rerun and `27.577 ms` slower than `new-goal.md`. Restored selected dprep=3 binary hash `7d03f302...` passed all nine focused smokes. |
| 2026-05-22 | Dprep warp-count candidate | Rejected `LLMK_SM120_DPREP_WARPS=2` as a trainer setting. It passed correctness and the focused attention benchmark was near selected timing, but end-to-end TinyStories first-three timing regressed, so dprep=3 remains selected. | `scratch/sm120_rounds/codex_sm120_dprep2_recovered_x3_20260522` passed all nine focused smokes and measured attention `787.510/2737.404 us`, but training steps were `2490.08, 2483.28, 2486.90 ms` with visible first3 `2486.753 ms`. That is `7.367 ms` slower than the latest selected direct rerun and `16.910 ms` slower than `new-goal.md`; restored selected dprep=3 binary hash `7d03f302...` passed all nine focused smokes. |
| 2026-05-22 | User-requested selected-stack rerun | Reran the exact selected `train-sm120.sh` path with the active `9fe90db1...` binary after confirming the GPU was idle. The run stayed in the recovered band but did not reproduce the user's pasted 2461 ms first-three band or beat `new-goal.md`. | `scratch/sm120_rounds/direct_train_sm120_9fe_user_rerun_verify9_x10_20260522/train-sm120.log` used CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar grad scale, maxconn1, and `-z 1`. x10 averaged `2485.469 ms`; first3 averaged `2479.387 ms`, which is `9.544 ms` slower than `new-goal.md` and `17.937 ms` slower than the user-pasted first3. Generated step-10 checkpoints were removed. |
| 2026-05-22 | Restored selected-stack direct rebaseline | Recorded the exact `train-sm120.sh` path after restoring the selected CUDA-kernel grad-zero build. The restored binary is back in the recovered band and effectively ties the prior reproduced first-three timing, but it does not beat `new-goal.md`. | `scratch/sm120_rounds/direct_train_sm120_9fe_restore_verify8_x10_20260522/train-sm120.log` used `train_gpt2cu` sha256 `9fe90db1...`, startup confirmed CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar grad scale, and maxconn1. x10 averaged `2485.751 ms`; first3 averaged `2478.147 ms`, only `0.004 ms` slower than the prior best reproduced first3 but `8.304 ms` slower than `new-goal.md`. Checkpoints were removed. |
| 2026-05-22 | Recovered-band CUDA runtime grad-zero retest | Rejected replacing the selected CUDA-kernel gradient zero route with CUDA runtime zeroing. This directly retests a benchmark conflict: runtime microbench rows show `cudaMemsetAsync` competitive, but the composed trainer route still does not produce a first-three improvement. | `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_recovered_x3_20260522` passed all nine focused CUDA smokes and startup confirmed `grad_zero_backend = CUDA runtime`, Torch C++ dresidual-zero, and host scalar grad scale. Trainer average was `2478.070 ms`, but the visible three-step average was `2479.853 ms`, slower than selected maxconn1 first3 `2478.143 ms` and `new-goal.md` target `2469.843 ms`. Checkpoints were removed; selected CUDA-kernel grad-zero build restored to `train_gpt2cu` sha256 `9fe90db1...`. |
| 2026-05-22 | Recovered-band maxconn16 retest | Rejected `CUDA_DEVICE_MAX_CONNECTIONS=16` again for the direct script. It was slightly better than the maxconn8 retest, but it still lost to the same-band maxconn1 run, so the runtime scheduling selection remains unchanged. | `scratch/sm120_rounds/direct_train_sm120_a9f_maxconn16_verify7_x10_20260522/train-sm120.log` used active binary `a9f1277e...`, startup confirmed CUDA-kernel grad-zero, Torch C++ dresidual-zero, and host scalar grad scale, and averaged `2485.517 ms` x10 with first3 `2478.910 ms`. Same-band maxconn1 remained better at `2484.309 ms` x10 / `2478.143 ms` first3. Checkpoints were removed. |
| 2026-05-22 | Recovered-band maxconn8 retest | Rejected `CUDA_DEVICE_MAX_CONNECTIONS=8` again for the direct script. This retest used the same active selected binary in the recovered runtime band, so it directly checks whether the earlier maxconn8 short-run signal should replace the script default. It should not. | `scratch/sm120_rounds/direct_train_sm120_a9f_maxconn8_verify6_x10_20260522/train-sm120.log` used active binary `a9f1277e...`, startup confirmed CUDA-kernel grad-zero, Torch C++ dresidual-zero, and host scalar grad scale, and averaged `2485.662 ms` x10 with first3 `2479.587 ms`. That is slower than the same-band maxconn1 direct rerun `2484.309 ms` x10 / `2478.143 ms` first3. Checkpoints were removed. |
| 2026-05-22 | Direct rerun with telemetry | Recorded three more exact `train-sm120.sh` reruns after the user's faster first-five observation. They show the active selected stack can recover from the severe slow band, but Codex still did not reproduce the user's 2461 ms first-three band, so this remains runtime-state work rather than a kernel promotion. | Telemetry run `scratch/sm120_rounds/codex_sm120_direct_train_telemetry_verify3_20260522/train-sm120.log` averaged `2485.844 ms` x10, first3 `2479.820 ms`; dmon showed mostly 99-100% SM, about 574-580 W, memory clock 13801 MHz, pclk mostly 2677-2707 MHz, and no power/thermal violation flags. Warm follow-up `scratch/sm120_rounds/direct_train_sm120_a9f_warm_verify4_x10_20260522.log` averaged `2489.568 ms`, first3 `2483.410 ms`. Fresh idle-GPU rerun `scratch/sm120_rounds/direct_train_sm120_a9f_verify5_x10_20260522/train-sm120.log` averaged `2484.309 ms`, first3 `2478.143 ms`. |
| 2026-05-22 | User-observed faster direct band | Recorded the user's pasted `train-sm120.sh` first-five timings as promising but not yet reproduced. It beats `new-goal.md` on first-three average, so it is now the highest-priority runtime state to reproduce with a captured artifact before claiming success. | User excerpt reported steps `2464.71/2456.65/2462.99/2461.49/2465.34 ms`, first3 avg `2461.450 ms`, first5 avg `2462.236 ms`, selected backend mix, normal `estimated maximum batch size: 73`, and normal losses. Codex rerun on active binary `a9f1277e...` did not reproduce it: captured `scratch/sm120_rounds/direct_train_sm120_a9f_verify2_x10_20260522.log` averaged `2511.285 ms`, first3 `2503.977 ms`. |
| 2026-05-22 | Precompute grad-scale plus maxconn8 composition | Rejected composing `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` with `CUDA_DEVICE_MAX_CONNECTIONS=8`. This was a new combined trainer test of the closest recovered-band device-side AdamW route with the fastest short-run scheduler signal, and the interaction is strongly negative. | Candidate `scratch/sm120_rounds/codex_sm120_precompute_grad_scale_maxconn8_x3_20260522` passed all nine focused CUDA smokes and validated, but x3 training averaged `2877.633 ms` with steps `5470.43/3290.21/2465.06 ms` and full memory pressure at startup. Checkpoints were removed; active selected-default binary after restore/check is `a9f1277e...`. |
| 2026-05-22 | Fresh selected-stack rebuild control | Recorded a fresh rebuild control for the selected promoted stack after the direct script entered a slow band. Rebuilding the selected stack recovered much of the slowdown, but it still did not beat `new-goal.md`. | `scratch/sm120_rounds/codex_sm120_fresh_rebuild_selected_control_x3_20260522` passed all nine focused smokes and validated at `2491.280 ms` x3 with CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar grad scale, and `CUDA_DEVICE_MAX_CONNECTIONS=1`; still `21.437 ms` slower than the target. |
| 2026-05-22 | Fresh exact `train-sm120.sh` slow-band control | Recorded another direct user-facing script run as a same-stack runtime regression point. The selected composed stack is active and the binary hash matches the restored promoted default, but the run is materially slower than the recovered exact x10 and the promoted proof. | Fresh `./train-sm120.sh` averaged `2575.607 ms` over x10 with startup confirming `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and BF16 on RTX 5090. Steps were `2572.43/2568.12/2558.55/2564.88/2586.59/2578.76/2582.10/2573.92/2583.25/2584.31 ms`; this is `85.814 ms` slower than the recovered exact x10 and `105.764 ms` slower than the `new-goal.md` first-three target. The generated checkpoint files were removed; active `train_gpt2cu` sha256 is `a55f2b0e...`. |
| 2026-05-22 | Harness zero-stage tracking plus precompute/zero0 combo | Rejected composing `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` with `TRAIN_ZERO_STAGE=0`. The harness now exposes `TRAIN_ZERO_STAGE` and records `train_zero_stage` in manifests so no-ZeRO experiments are tracked like other candidate rounds. | Candidate `scratch/sm120_rounds/codex_sm120_precompute_grad_scale_zero0_x3_20260522` passed all nine focused CUDA smokes and averaged `2488.396 ms` over three TinyStories steps with `grad_scale_backend = precomputed device AdamW scalar` and `zero_stage = 0`. That is `9.002 ms` slower than the recovered selected-stack x3 and `18.553 ms` slower than the `new-goal.md` target. Promoted-default binaries were restored in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_precompute_zero0_20260522`; all nine smokes passed, restored `train_gpt2cu` sha256 `a55f2b0e...`. |
| 2026-05-22 | Promoted stack TK backward N96 tile selector | Rejected `LLMK_SM120_BACKWARD_N96=1` on top of the promoted stack. The selector is correctness-clean, but the real trainer regresses versus the recovered selected-stack control and stays well above the target. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_backward_n96_x3_20260522` passed all nine focused CUDA smokes, captured native benchmarks, and averaged `2487.329 ms` over three TinyStories steps. That is `7.935 ms` slower than the recovered selected-stack x3 (`2479.394 ms`) and `17.486 ms` slower than the `new-goal.md` target. Promoted-default binaries were rebuilt afterward in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_backward_n96_20260522`; all nine smokes passed, restored `train_gpt2cu` sha256 `79bc5e2e...`. |
| 2026-05-22 | Single-GPU ZeRO stage setting | Rejected changing `train-sm120.sh` from `-z 1` to `-z 0`. The one-process no-ZeRO path is not faster in the recovered band. | Direct probe `log124M/5090_S_zero0_probe_x3_20260522` averaged `2480.059 ms` with steps `2484.12/2478.42/2481.70 ms`, normal losses, and startup confirming CUDA-kernel grad-zero plus Torch C++ dresidual-zero. That is `0.665 ms` slower than the recovered selected-stack x3 (`2479.394 ms`) and `10.216 ms` slower than the `new-goal.md` target. Checkpoint files were removed. |
| 2026-05-22 | Promoted stack precomputed AdamW grad-scale recovered-band retest | Rejected `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` again after retesting it in the recovered fast band. The candidate is correctness-clean and stays fast, but it does not beat the same-band selected-stack control or the `new-goal.md` target. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_precompute_grad_scale_recovered_x3_20260522` passed all nine focused CUDA smokes and averaged `2480.090 ms` over three TinyStories steps with `grad_scale_backend = precomputed device AdamW scalar`. It is `0.696 ms` slower than the recovered selected-stack x3 (`2479.394 ms`) and `10.247 ms` slower than the `new-goal.md` target (`2469.843 ms`). Promoted-default binaries were rebuilt afterward in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_precompute_recovered_20260522`; all nine smokes passed, with restored `train_gpt2cu` sha256 `4f4ea367...`. |
| 2026-05-22 | Exact `train-sm120.sh` recovered-band rerun | Recorded that the exact user-facing script no longer reproduces the earlier slow band. This supports treating the slowdown as runtime-band variability in the same selected mixed CUDA/Torch stack, not as a rejected kernel candidate accidentally left enabled. | `./train-sm120.sh` averaged `2489.794 ms` over x10 with steady steps `2483.67` to `2495.52 ms`; startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, BF16 on RTX 5090. This is `51.839 ms` faster than the degraded direct-script control (`2541.633 ms`) and `3.339 ms` faster than stable x10 (`2493.133 ms`), but still `0.731 ms` slower than promoted direct proof and `19.951 ms` slower than the `new-goal.md` first-three target. |
| 2026-05-22 | Current selected-stack telemetry probe | Recorded a short recovered-band current-stack control with GPU telemetry. This shows the selected stack can still run in the fast band when the GPU is fully loaded and not power/thermal throttling. | Direct current-binary x3 run in `log124M/5090_S_runtime_telemetry_probe_x3_20260522` averaged `2479.394 ms`; dmon showed `99-100%` SM utilization, `574-575 W`, graphics clocks around `2692-2775 MHz`, memory clock `13801 MHz`, and `pviol=0` / `tviol=0` during training. The short run is `62.239 ms` faster than the degraded direct-script control, but still `9.551 ms` slower than the `new-goal.md` target. |
| 2026-05-22 | Historical antigravity binary probe | Rejected the separate `train_gpt2cu_antigravity` binary as an explanation for the faster `new-goal.md` band. It is not source-promotable and is much slower than every relevant current target. | Direct x3 TinyStories run in `log124M/5090_S_antigravity_probe_x3_20260522` averaged `3312.640 ms` with steps `3304.33/3309.41/3315.87 ms`, final val loss `10.609902`, and binary sha256 `948b57bb...`. That is `842.797 ms` slower than the `new-goal.md` target (`2469.843 ms`), `823.578 ms` slower than promoted direct proof (`2489.062 ms`), and `771.007 ms` slower than the current direct-script regression control (`2541.633 ms`). |
| 2026-05-22 | Promoted backward sync opt-out | Rejected `LLMK_SM120_DISABLE_BACKWARD_STREAM_SYNC` on top of the promoted SM120 fast trainer. The older device-wide sync slightly improves the current degraded script band, but it does not recover the stable/promoted training speed and does not beat the stream-scoped sync evidence that was promoted. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522` passed all nine focused CUDA smokes and averaged `2535.141 ms` over 3 TinyStories steps. That is `6.492 ms` faster than the current direct-script regression control (`2541.633 ms`) but `46.078 ms` slower than promoted direct proof (`2489.062 ms`) and `42.008 ms` slower than stable x10. Promoted-default binaries were restored in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_disable_backward_stream_sync_20260522`; all nine focused CUDA smokes passed, with restored `train_gpt2cu` sha256 `c8934ab4...`. |
| 2026-05-22 | Promoted fcproj direct-cuBLAS plus maxconn8 | Rejected composing `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` with `CUDA_DEVICE_MAX_CONNECTIONS=8`. This was a real combined-kernel-stack trainer test, not just a scorecard readout, and the interaction regressed badly. | Candidate `scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_maxconn8_x3_20260522` passed all nine focused CUDA smokes and captured native benchmarks, but TinyStories x3 averaged `2678.119 ms`. That is `189.057 ms` slower than promoted direct proof (`2489.062 ms`), `184.986 ms` slower than stable x10, and `194.668 ms` slower than maxconn8 alone. Promoted-default binaries were restored in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_cublas_dinp_fcproj_maxconn8_20260522`; all nine focused CUDA smokes passed, with restored `train_gpt2cu` sha256 `960c820c...`. |
| 2026-05-22 | Promoted stack without direct-cuBLAS backward selector | Rejected promoting `LLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM`. Disabling the older direct-cuBLAS backward selector improves the current degraded band, but it still does not beat the stable/promoted training baselines, so it is only a current-band regression clue. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_disable_cublas_bwd_x3_20260522` passed all nine focused CUDA smokes and averaged `2521.217 ms` over 3 TinyStories steps. That is `20.416 ms` faster than the current direct-script regression control (`2541.633 ms`) but `32.154 ms` slower than promoted direct proof (`2489.062 ms`) and `28.084 ms` slower than stable x10. Promoted-default binaries were restored in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_disable_cublas_bwd_20260522`; all nine focused CUDA smokes passed, with restored `train_gpt2cu` sha256 `da47e81e...`. |
| 2026-05-22 | Promoted stack cuBLASLt heuristic-results=1 | Rejected forcing `LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS=1` on top of the promoted SM120 fast trainer. This tested whether the current wider cuBLASLt heuristic search was causing the promoted-stack slowdown, but the actual trainer got slower. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_cublaslt_heur1_x3_20260522` passed all nine focused CUDA smokes and captured native benchmarks, but TinyStories x3 averaged `2552.864 ms`. That is `63.802 ms` slower than promoted direct `2489.062 ms`, `59.731 ms` slower than stable x10, and `11.232 ms` slower than the current direct-script regression control. Promoted-default binaries were rebuilt afterward without the flag in `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_heur1_20260522`, and all nine focused CUDA smokes passed; restored `train_gpt2cu` sha256 `240bb78c...`. |
| 2026-05-22 | Promoted stack epilogue/probe control | Kept the default cuBLASLt fused-epilogue heuristic selection and recorded a fresh promoted-stack control. The focused epilogue probe did not show a stable alternate algorithm edge, and the restored selected stack still runs in the current slow band. | `scratch/sm120_rounds/codex_sm120_cublaslt_epilogue_probe_20260522/bench_sm120_cublaslt_epilogue_algos.log` shows `fc fwd+GeLU` default index 0 at `1512.053 us` versus index 1 at `1557.621 us`; `fcproj dInp+dGeLU` index 1 initially measured `1901.317 us`, but default index 0 retested faster at `1871.051 us`. Control `scratch/sm120_rounds/codex_sm120_promoted_default_after_epilogue_probe_x3_20260522` passed all nine smokes and averaged `2594.888 ms`, with selected backends still active. |
| 2026-05-22 | Direct `train-sm120.sh` slowdown check | Verified the current user-facing script directly, without changing source or rebuilding. The selected composed stack is active, but the script is currently slower than the promoted proof, so this is a regression-control datapoint rather than a new baseline. | `scratch/sm120_rounds/codex_sm120_direct_train_sm120_current_no_rebuild_x10_20260522/train-sm120.log` averaged `2541.633 ms`; startup printed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, `gelu_fusion = 1`, BF16 on RTX 5090. The run is `52.571 ms` slower than the promoted direct proof (`2489.062 ms`) and `48.500 ms` slower than stable x10 (`2493.133 ms`). |
| 2026-05-22 | Promoted stack attention bwd32 tile | Rejected `LLMK_SM120_ATTN_BWD_BLOCK=32` on top of the promoted SM120 fast trainer. It is faster than the latest slow selected-stack control, but it does not beat the promoted proof or stable x10 baseline and the focused attention benchmark gets worse. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522` passed all nine focused CUDA smokes. Native `bench_sm120_attention.log` measured forward/backward `830.444/3074.490 us` versus selected packed-QKV forward/backward `784.691/2716.901 us`; TinyStories x3 averaged `2502.501 ms`, `13.439 ms` slower than promoted direct `2489.062 ms` and `9.368 ms` slower than stable x10. The promoted default binary was rebuilt afterward without the bwd32 flag and the restore smoke suite passed, leaving `train_gpt2cu` sha256 `9940b759...`. |
| 2026-05-22 | Promoted stack attention fwd16 tile | Rejected `LLMK_SM120_ATTN_FWD_BLOCK=16` on top of the promoted SM120 fast trainer. The candidate is correctness-clean and somewhat faster than the latest slow selected-stack control, but the focused attention benchmark gets worse and end-to-end training remains far behind the promoted proof and stable x10 baseline. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_attn_fwd16_x3_20260522` passed all nine focused CUDA smokes. Native `bench_sm120_attention.log` measured forward/backward `1055.652/2747.087 us` versus the selected packed-QKV forward/backward `784.691/2716.901 us`; TinyStories x3 averaged `2531.237 ms`, `42.175 ms` slower than promoted direct `2489.062 ms`. The promoted default binary was rebuilt afterward without the fwd16 flag and the restore smoke suite passed, leaving `train_gpt2cu` sha256 `930321ac...`. |
| 2026-05-22 | Promoted stack post-scheduler control | Recorded the selected `train-sm120.sh` stack as a current regression-control after the scheduler sweep. The stack is still CUDA-kernel grad-zero, Torch C++ dresidual-zero, host-scalar AdamW grad scale, fused GELU, and `CUDA_DEVICE_MAX_CONNECTIONS=1`; it is slower now because the same stack showed step-time spikes, not because a rejected scheduler candidate was left selected. | Control `scratch/sm120_rounds/codex_sm120_promoted_default_post_sched_x10_20260522` passed all nine focused CUDA smokes, startup confirmed the selected mixed backend stack, and TinyStories x10 averaged `2543.390 ms` with spikes on steps 2-3 and 9-10. That is `54.328 ms` slower than promoted direct `2489.062 ms` and `50.257 ms` slower than stable x10; checkpoint files were cleaned and `train_gpt2cu` is sha256 `d1db5331...`. |
| 2026-05-22 | Promoted stack maxconn16 scheduling | Rejected `CUDA_DEVICE_MAX_CONNECTIONS=16` for `train-sm120.sh` after the required x10 gate. The x3 smoke was fast enough to shortlist, but the longer run regressed with early and late spikes and lost to both the promoted direct proof and stable x10 baseline. | Candidate x3 `scratch/sm120_rounds/codex_sm120_promoted_maxconn16_x3_20260522` passed all nine focused CUDA smokes and averaged `2485.113 ms`, `3.949 ms` faster than promoted direct `2489.062 ms`. The x10 gate `scratch/sm120_rounds/codex_sm120_promoted_maxconn16_x10_20260522` also passed correctness but averaged `2525.956 ms`, `36.894 ms` slower than promoted direct and `32.823 ms` slower than stable x10; checkpoint files were cleaned and the runtime-only x10 build is `train_gpt2cu` sha256 `f6e6aa87...`. |
| 2026-05-22 | Promoted stack maxconn8 scheduling | Rejected `CUDA_DEVICE_MAX_CONNECTIONS=8` for `train-sm120.sh` after the required x10 gate. The x3 smoke was genuinely fast, but the longer run developed late spikes and lost to both the promoted direct proof and stable x10 baseline. | Candidate x3 `scratch/sm120_rounds/codex_sm120_promoted_maxconn8_x3_20260522` passed all nine focused CUDA smokes and averaged `2483.451 ms`, `5.611 ms` faster than promoted direct `2489.062 ms`. The x10 gate `scratch/sm120_rounds/codex_sm120_promoted_maxconn8_x10_20260522` also passed correctness but averaged `2509.947 ms`, `20.885 ms` slower than promoted direct and `16.814 ms` slower than stable x10; checkpoint files were cleaned and `train_gpt2cu` remained a promoted-stack runtime-only build, sha256 `6d2b3a55...`. |
| 2026-05-22 | Promoted stack maxconn4 scheduling | Rejected `CUDA_DEVICE_MAX_CONNECTIONS=4` for `train-sm120.sh`. It improves over the already-rejected maxconn2 setting, but it is still materially slower than both the promoted direct proof and stable x10 baseline, so the selected runtime setting remains maxconn1. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_maxconn4_x3_20260522` passed all nine focused CUDA smokes, startup confirmed CUDA-kernel grad-zero and Torch C++ dresidual-zero, and TinyStories x3 averaged `2531.798 ms` with normal losses. That is `42.735 ms` slower than promoted direct `2489.062 ms` and `38.665 ms` slower than stable x10; no source or binary restore was required because this was a runtime-only scheduling retest and `train_gpt2cu` remained sha256 `3a7b18eb...`. |
| 2026-05-22 | Promoted stack device AdamW grad scale | Rejected adding `LLMK_SM120_DEVICE_GRAD_SCALE_ADAMW` on top of the promoted SM120 fast trainer. The direct device-scalar route is slower than the already-rejected precomputed scalar route and does not improve the selected host-scalar AdamW update path. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_device_grad_scale_adamw_x3_20260522` passed all nine focused CUDA smokes, startup confirmed `grad_scale_backend = device AdamW scalar`, and TinyStories x3 averaged `2528.446 ms` with normal losses. That is `39.384 ms` slower than promoted direct `2489.062 ms` and `35.313 ms` slower than stable x10; promoted-default binaries were rebuilt afterward to `train_gpt2cu` sha256 `68e17456...` and the restored smoke suite passed. |
| 2026-05-22 | Promoted stack async grad-norm copy | Rejected adding `LLMK_SM120_ASYNC_GRAD_NORM_COPY` on top of the promoted SM120 fast trainer. The stream-scoped async scalar readback plus stream sync is still slower when composed with the selected CUDA-kernel grad-zero, Torch C++ residual-zero, dprep3, block1024, LayerNorm bwd1, and maxconn1 stack. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_async_grad_norm_copy_x3_20260522` passed all nine focused CUDA smokes, startup confirmed CUDA-kernel grad-zero and Torch C++ dresidual-zero, and TinyStories x3 averaged `2508.091 ms` with normal losses. That is `19.029 ms` slower than promoted direct `2489.062 ms` and `14.958 ms` slower than stable x10; promoted-default binaries were rebuilt afterward to `train_gpt2cu` sha256 `30e73fb9...` and the restored smoke suite passed. |
| 2026-05-22 | Promoted stack fcproj direct-cuBLAS x10 gate | Rejected promoting `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` after the required x10 confirmation. The first x10 was promising and beat the same-session promoted default, but the repeat x10 regressed with late-step spikes, so it is not a stable faster trainer selection. | First gate `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_fcproj_x10_20260522` passed all nine focused CUDA smokes and averaged `2490.019 ms`, which is `3.114 ms` faster than stable x10 but `0.957 ms` slower than promoted direct proof. Same-session promoted default averaged `2533.484 ms`. Confirmation `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_fcproj_confirm_x10_20260522` averaged `2548.056 ms`; promoted-default binaries were rebuilt afterward to `train_gpt2cu` sha256 `1edf1131...` and the restored smoke suite passed. |
| 2026-05-21 | Promoted stack fcproj direct-cuBLAS dInput | Rejected adding `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` on top of the promoted SM120 fast trainer. This was the closest remaining promoted-stack dInput selector retest, but it still does not beat either the promoted direct proof or the stable x10 baseline. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_fcproj_x3_20260521` passed all nine focused CUDA smokes, startup confirmed CUDA-kernel grad-zero and Torch C++ dresidual-zero, and TinyStories x3 averaged `2493.911 ms` with normal losses. That is `4.849 ms` slower than promoted direct `2489.062 ms` and `0.778 ms` slower than stable x10; promoted-default binaries were rebuilt afterward to `train_gpt2cu` sha256 `6c6f8a14...` and the restored smoke suite passed. |
| 2026-05-21 | Promoted stack qkv direct-cuBLAS dInput | Rejected adding `LLMK_SM120_USE_CUBLAS_DINP_QKV` on top of the promoted SM120 fast trainer. This retested the qkv dInput selector in the selected CUDA-kernel grad-zero, Torch C++ residual-zero, dprep3, block1024, LayerNorm bwd1, and maxconn1 composition; it still does not produce a training-speed win. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_qkv_x3_20260521` passed all nine focused CUDA smokes, startup confirmed CUDA-kernel grad-zero and Torch C++ dresidual-zero, and TinyStories x3 averaged `2607.288 ms` with normal losses. That is `118.226 ms` slower than promoted direct `2489.062 ms` and `114.155 ms` slower than stable x10; promoted-default binaries were rebuilt afterward to `train_gpt2cu` sha256 `3a268f8f...` and the restored smoke suite passed. |
| 2026-05-21 | Promoted stack cuBLASLt workspace/search | Rejected adding `LLMK_SM120_CUBLASLT_WORKSPACE_MB=256` and `LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS=16` on top of the promoted SM120 fast trainer. This retested the older cuBLASLt workspace/search tuning knob in the selected CUDA-kernel grad-zero, Torch C++ residual-zero, dprep3, block1024, LayerNorm bwd1, and maxconn1 composition; it still does not produce a training-speed win. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_cublaslt_ws256_heur16_x3_20260521` passed all nine focused CUDA smokes, startup confirmed CUDA-kernel grad-zero and Torch C++ dresidual-zero, and TinyStories x3 averaged `2603.521 ms` with normal losses. That is `114.459 ms` slower than promoted direct `2489.062 ms` and `110.388 ms` slower than stable x10; promoted-default binaries were rebuilt afterward to `train_gpt2cu` sha256 `a55413ce...` and the restored smoke suite passed. |
| 2026-05-21 | Promoted stack cuBLASLt plan cache | Rejected adding `LLMK_SM120_CACHE_CUBLASLT_PLANS` on top of the promoted SM120 fast trainer. This retested the older trainer-wide cuBLASLt plan-cache knob in the current CUDA-kernel grad-zero, Torch C++ residual-zero, dprep3, block1024, LayerNorm bwd1, and maxconn1 composition; it still does not produce a training-speed win. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_cublaslt_plan_cache_x3_20260521` passed all nine focused CUDA smokes, startup confirmed CUDA-kernel grad-zero and Torch C++ dresidual-zero, and TinyStories x3 averaged `2597.241 ms` with normal losses. That is `108.178 ms` slower than promoted direct `2489.062 ms` and `104.108 ms` slower than stable x10; promoted-default binaries were rebuilt afterward to `train_gpt2cu` sha256 `d06861f6...` and the restored smoke suite passed. |
| 2026-05-21 | Promoted stack global-norm block256 | Rejected adding `LLMK_SM120_GLOBAL_NORM_BLOCK_SIZE=256` on top of the promoted SM120 fast trainer. This retested the earlier focused global-norm block-size signal in the real trainer composition; it improved the current slow telemetry band but did not beat the promoted direct proof or stable x10 baseline. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_global_norm_block256_x3_20260521` passed all nine focused CUDA smokes, startup confirmed CUDA-kernel grad-zero and Torch C++ dresidual-zero, and TinyStories x3 averaged `2590.486 ms` with normal losses. That is `101.423 ms` slower than promoted direct `2489.062 ms` and `97.353 ms` slower than stable x10; promoted-default binaries were rebuilt afterward to `train_gpt2cu` sha256 `2a0804fd...` and the restored smoke suite passed. |
| 2026-05-21 | Promoted memory store-policy sweep | Keep the promoted native memory wrapper on `LLMK_SM120_MEMORY_STORE_POLICY=1`. The two remaining store policies were tested on top of the promoted SM120 fast trainer because they directly affect the CUDA-kernel grad-zero path; both regressed training badly. | `scratch/sm120_rounds/codex_sm120_promoted_memory_store_cg_x3_20260521` used `LLMK_SM120_MEMORY_STORE_POLICY=2`, passed all nine focused CUDA smokes, and averaged `2592.754 ms`. `scratch/sm120_rounds/codex_sm120_promoted_memory_store_default_x3_20260521` used `LLMK_SM120_MEMORY_STORE_POLICY=0`, passed all nine smokes, and averaged `2593.304 ms`. Both are about `104 ms` slower than promoted direct `2489.062 ms`; promoted-default binaries were rebuilt afterward to `train_gpt2cu` sha256 `b36d91d...` and the restored smoke suite passed. |
| 2026-05-21 | Promoted stack cuBLAS dInput plus TK dGELU | Rejected adding both direct-cuBLAS dInput selectors and `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` on top of the promoted SM120 fast trainer. This was a composed trainer test of the prior fastest-row dInput/dGELU idea against the selected CUDA-kernel grad-zero, Torch C++ residual-zero, dprep3, block1024, LayerNorm bwd1, and maxconn1 stack; the short-run edge did not survive x10. | Candidate x3 `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_tk_dgelu_x3_20260521` passed all nine focused CUDA smokes and averaged `2491.288 ms`, so it was gated. The x10 gate `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_tk_dgelu_x10_20260521` averaged `2508.456 ms`, `19.394 ms` slower than promoted direct `2489.062 ms`; promoted-default binaries were rebuilt afterward to `train_gpt2cu` sha256 `8254ee2d...` and the restored smoke suite passed. |
| 2026-05-21 | Promoted stack LibTorch grad-zero plus TK dGELU | Rejected combining LibTorch grad-zero with `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` on top of the promoted SM120 fast trainer. This was a direct composition of the remaining trainer-callable Torch memory route with the best current-band add-on, and the interaction is negative. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_libtorch_grad_zero_tk_dgelu_x3_20260521` passed all nine focused CUDA smokes, startup confirmed Torch C++ grad-zero and Torch C++ dresidual-zero, and TinyStories x3 averaged `2607.654 ms` with normal losses. That is `104.895 ms` slower than TK dGELU alone and `118.592 ms` slower than promoted direct `2489.062 ms`; promoted-default binaries were rebuilt afterward without the candidate flags and the restored smoke suite passed. |
| 2026-05-21 | Promoted stack LibTorch grad-zero retest | Rejected replacing the promoted CUDA-kernel grad-zero route with LibTorch grad-zero while keeping the rest of the promoted SM120 fast trainer. This directly tests the remaining trainer-callable Torch memory route in the promoted stack, and it is still not a real training-speed promotion. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_libtorch_grad_zero_x3_20260521` passed all nine focused CUDA smokes, startup confirmed Torch C++ grad-zero and Torch C++ dresidual-zero, and TinyStories x3 averaged `2593.987 ms` with normal losses. That is `16.994 ms` faster than the current slow maxconn1 reference but `104.925 ms` slower than promoted direct `2489.062 ms`; promoted-default binaries were rebuilt afterward without the LibTorch grad-zero flag and the restored smoke suite passed. |
| 2026-05-21 | Promoted stack precomputed AdamW grad scale | Rejected adding `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` on top of the promoted SM120 fast trainer. It is a valid composed trainer test of the remaining AdamW grad-scale route with CUDA-kernel grad-zero, Torch C++ residual-zero, dprep3, block1024, LayerNorm bwd1, and maxconn1, but it does not produce a real speed promotion. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_precompute_grad_scale_x3_20260521` passed all nine focused CUDA smokes, startup confirmed CUDA-kernel grad-zero, Torch C++ dresidual-zero, and precomputed device AdamW scalar, and TinyStories x3 averaged `2604.345 ms` with normal losses. That is only `6.636 ms` faster than the current slow maxconn1 reference and `115.283 ms` slower than promoted direct `2489.062 ms`; promoted-default binaries were rebuilt afterward without the flag and the restored smoke suite passed. |
| 2026-05-21 | Promoted stack TK dGELU plus classifier exp2 | Rejected combining `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` and `LLMK_SM120_CLASSIFIER_EXP2` on top of the promoted SM120 fast trainer. The interaction does not preserve the TK dGELU current-band improvement and is not close to a real training-speed promotion. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_tk_dgelu_classifier_exp2_x3_20260521` passed all nine focused CUDA smokes, startup confirmed CUDA-kernel grad-zero and Torch C++ dresidual-zero, and TinyStories x3 averaged `2609.867 ms` with normal losses. That is `107.109 ms` slower than TK dGELU alone and `120.805 ms` slower than promoted direct `2489.062 ms`; promoted-default binaries were rebuilt afterward without the flags and the restored smoke suite passed. |
| 2026-05-21 | Promoted stack TK dGELU | Rejected adding `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` to the promoted SM120 fast trainer. This was the best current-band add-on tested this turn, but it still loses to the promoted direct proof and stable x10 baseline. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_tk_dgelu_x3_20260521` passed all nine focused CUDA smokes, startup confirmed CUDA-kernel grad-zero and Torch C++ dresidual-zero, and TinyStories x3 averaged `2502.759 ms` with normal losses. That is `108.223 ms` faster than the current slow maxconn1 telemetry reference but `13.696 ms` slower than promoted direct `2489.062 ms`; promoted-default binaries were rebuilt afterward without the flag and the restored smoke suite passed. |
| 2026-05-21 | Promoted stack classifier exp2 | Rejected adding `LLMK_SM120_CLASSIFIER_EXP2` to the promoted SM120 fast trainer. It improved the current slow band, but not enough to matter against the promoted proof or stable x10 baseline, so this is not a real training-speed promotion. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_classifier_exp2_x3_20260521` passed all nine focused CUDA smokes, startup confirmed CUDA-kernel grad-zero and Torch C++ dresidual-zero, and TinyStories x3 averaged `2580.789 ms` with normal losses. That is `30.192 ms` faster than the current slow maxconn1 telemetry reference but `91.727 ms` slower than promoted direct `2489.062 ms`; promoted-default binaries were rebuilt afterward without the flag and the restored smoke suite passed. |
| 2026-05-21 | Promoted stack cuBLAS dInput revisit | Rejected adding the attention-projection and MLP-up direct-cuBLAS dInput selectors to the promoted SM120 fast trainer. This was a composed trainer test on top of the selected CUDA-kernel grad-zero, Torch C++ residual-zero, dprep3, memory block1024, LayerNorm bwd1, and maxconn1 stack, not an isolated microbench scoreboard row. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_attproj_fc_x3_20260521` passed all nine focused CUDA smokes and trained with normal losses, but averaged `2618.905 ms` over 3 TinyStories steps. It was `7.924 ms` slower than the current slow maxconn1 reference `2610.981 ms` and `129.843 ms` slower than the promoted direct proof `2489.062 ms`; promoted-default binaries were rebuilt afterward without the selector flags, restoring `train_gpt2cu` sha256 `d061c1...`. |
| 2026-05-21 | Promoted scheduling and residual ablations | Keep the promoted `CUDA_DEVICE_MAX_CONNECTIONS=1` script default and Torch C++ residual clear. In the current slow band, unsetting max connections looked slightly better on x3, but the x10 gate regressed; `CUDA_DEVICE_MAX_CONNECTIONS=2` and CUDA runtime dresidual-zero were also slower. | No-maxconn x3 `scratch/sm120_rounds/codex_sm120_promoted_no_maxconn_x3_20260521` averaged `2597.961 ms`, but x10 `scratch/sm120_rounds/codex_sm120_promoted_no_maxconn_x10_20260521` averaged `2631.929 ms`. `CUDA_DEVICE_MAX_CONNECTIONS=2` x3 averaged `2613.712 ms`. CUDA runtime dresidual-zero x3 averaged `2624.266 ms`. All four candidates passed all nine focused CUDA smokes and preserved normal losses. |
| 2026-05-21 | Current direct slow band | Treat the current `train-sm120.sh` slowdown as a runtime/operating-state regression reference, not a silent switch away from the promoted components. The direct script is still exporting `CUDA_DEVICE_MAX_CONNECTIONS=1` and the restored binary still reports the promoted CUDA-grad-zero plus Torch-dresidual-zero backends. | Direct telemetry-backed run `scratch/train-sm120-promoted-slow-dmon-20260521.log` averaged `2610.981 ms` with steps around `2600-2621 ms`. Startup confirmed `grad_zero_backend = CUDA kernel`, `dresidual_zero_backend = Torch C++`, `grad_scale_backend = host scalar`, and `gelu_fusion = 1`. `scratch/dmon-train-sm120-promoted-slow-20260521.log` shows `99-100%` SM utilization, `573-575 W`, pclk about `2707-2782 MHz`, memory clock `13801 MHz`, and `pviol=0`, `tviol=0`. |
| 2026-05-21 | Promoted stack plus wide bias-add | Rejected adding `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024` on top of the promoted SM120 fast trainer stack. This was a composed training test against the promoted direct-script baseline, not a standalone bias-add scoreboard decision, and it regressed enough on x3 that no x10 gate is justified. | Candidate `scratch/sm120_rounds/codex_sm120_promoted_bias_wide1024_x3_20260521` used CUDA-kernel grad-zero, Torch C++ dresidual-zero, dprep=3, memory block1024, LayerNorm bwd1, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and the wide-bias block override. All nine focused CUDA smokes passed, startup confirmed `grad_zero_backend = CUDA kernel` and `dresidual_zero_backend = Torch C++`, but TinyStories x3 averaged `2608.036 ms`, slower than promoted direct `2489.062 ms` by `118.973 ms`. Promoted-default binaries were rebuilt afterward without the wide-bias flag; restored `train_gpt2cu` sha256 `45a26966...`. |
| 2026-05-21 | Promoted SM120 fast trainer stack | Promote the composed SM120 stack: CUDA-kernel gradient zeroing, Torch C++ residual zeroing, attention `LLMK_SM120_DPREP_WARPS=3`, memory zero block size `1024`, LayerNorm backward one-block-per-SM, and `CUDA_DEVICE_MAX_CONNECTIONS=1` for `train-sm120.sh`. This is the first composed stack in this pass to beat the stable x10 baseline twice and then verify through the direct user-facing script. | Harness x10 `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x10_20260521` averaged `2490.940 ms`; confirmation x10 `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_confirm_x10_20260521` averaged `2490.432 ms`; direct `./train-sm120.sh` after source promotion averaged `2489.062 ms`, beating the prior stable `2493.133 ms` baseline by `4.071 ms`. |
| 2026-05-21 | cuBLAS dInput + dprep combination | Rejected the combined `LLMK_SM120_USE_CUBLAS_DINP_ATTPROJ`, `LLMK_SM120_USE_CUBLAS_DINP_FC`, and `LLMK_SM120_DPREP_WARPS=3` trainer route. This was the direct composition of the closest x10 cuBLAS dInput selector near-miss with the attention dprep=3 near-miss, but it did not compose into a training win. | Target-context round `scratch/sm120_rounds/codex_sm120_combo_cublas_dinp_attproj_fc_dprep3_x3_20260521` passed all nine focused CUDA smokes and trained with normal losses, but averaged `2543.476 ms` over 3 TinyStories steps, `50.343 ms` slower than stable x10 `2493.133 ms`. Default binaries were rebuilt afterward and `train_gpt2cu` restored to `cbcf72...`. |
| 2026-05-21 | Live CUDA scheduling controls | Do not promote `CUDA_DEVICE_MAX_CONNECTIONS=1` by itself or with the all-native stack. It only became promotable when combined with the later CUDA-grad-zero + Torch-dresidual-zero + dprep3 + LayerNorm bwd1 + block1024 stack above. | Restored default binary `cbcf72...` live control averaged `2527.266 ms` x3. `CUDA_DEVICE_MAX_CONNECTIONS=1` improved same-session x3 to `2495.607 ms`, but x10 averaged `2510.431 ms`, slower than stable `2493.133 ms`. `CUDA_DEVICE_MAX_CONNECTIONS=32` averaged `2512.792 ms` x3. The best all-native stack plus maxconn1 passed all nine focused CUDA smokes but averaged `2506.429 ms` x3, slower than maxconn1 default and the stable baseline. Later composed-stack evidence supersedes this as a standalone scheduling decision. |
| 2026-05-21 | Objective contract doc guard | Treat `optimise-goal.md` as part of the current SM120 selection replay contract. The goal file now names the current `43/43` Torch objective benchmark coverage from `dev/sm120_objective_contract.py`, and the audit fails if that count drifts back to the stale `42/42` wording. | Added `optimise-goal.md` to the default `--doc-path` set in `dev/audit_sm120_optimization_goal.py`, extended `audit_docs()` with the `objective Torch row count docs` check, and regenerated `scratch/sm120_rounds/current-sm120-audit.{json,md}`. The latest audit reports `checks=131`, `torch_objective_rows=43`, `active_promotions=0`, and an empty stale-or-missing set for `optimise-goal.md`. |
| 2026-05-21 | FC-only cuBLAS dInput selector | Keep the GPT-2 MLP-up dInput row on the current cuBLASLt trainer route. The new `LLMK_SM120_USE_CUBLAS_DINP_FC` A/B hook isolates the MLP-up row from the previously bundled attention-projection selector, and the isolated x10 stability gate still regresses. | Default build passed after adding behavior-preserving hooks `LLMK_SM120_USE_CUBLAS_DINP_ATTPROJ` and `LLMK_SM120_USE_CUBLAS_DINP_FC`. FC-only target-context round `scratch/sm120_rounds/codex_sm120_round_cublas_dinp_fc_only_20260521` passed all native smokes and 3 TinyStories steps at `avg_ms=2490.977`; x10 `scratch/sm120_rounds/codex_sm120_round_cublas_dinp_fc_only_x10_20260521` also passed all gates but regressed to `avg_ms=2504.399` versus current native x10 `2493.133 ms`. The default binaries were rebuilt without the candidate flag. |
| 2026-05-21 | Bias-add wide-block trainer check | Rejected `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024` as a default source change after running it through the full native SM120 round harness. The candidate is correctness-clean and trainer-callable, but the focused runtime row did not reproduce the earlier one-off wide-row edge, and the 3-step TinyStories timing is not a stable-best result. | Target-context round `scratch/sm120_rounds/codex_sm120_round_bias_add_wide1024_20260521` built with `EXTRA_NVCC_FLAGS=-DLLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024`, passed all nine CUDA correctness smokes, benchmark validation, and 3 TinyStories steps at `avg_ms=2489.491`. The same round measured `bias_add BT=65536 OC=768/3072` at `101.293/536.232 us`, so it does not reproduce the earlier `527.753 us` `OC=3072` candidate row; it is also slower than the better prior 3-step stream-sync candidate round at `2483.598 ms`. The default `train_gpt2cu` binary was rebuilt without the candidate flag. |
| 2026-05-21 | Profiler-only memory rows | Keep the logits-sized memset row and logits/hidden-sized device-copy rows as profiler/runtime evidence, not trainer integration backlog. The LibTorch logits copy remains the fastest observed profiler row, but the current GPT-2 trainer has no matching device-to-device copy call-site to promote. | Source search of `train_gpt2.cu` found host/device copies for inputs, targets, mean loss, and sampling logits, plus real zeroing paths for losses, grads, residual scratch, AdamW state, and attention scratch, but no logits-sized memset or logits/hidden-sized D2D copy call-site. `dev/sm120_promotion_decisions.json` now marks those three rows `profiler_only_runtime_row`; regenerated `current-sm120-selection.md` reports `profiler_only_runtime_row: 3` in the resolved decision statuses and resolves `cuda_copy_d2d logits_elems=3296722944` and `cuda_memset logits_elems=3296722944` away with no current trainer call-site. |
| 2026-05-21 | LibTorch dresidual-zero trainer route | Keep the backward residual stream clear on CUDA runtime by default. The Torch/Torch C++ `cuda_memset hidden_elems=50331648` row had a sub-percent operator edge, and the opt-in trainer integration now exists, but the x10 TinyStories stability check regressed versus the current native selection. | Added `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1` plus a cached BF16 `from_blob` handle over `model->acts.scratch_btc`; startup prints `dresidual_zero_backend \| Torch C++` while `grad_zero_backend` remains CUDA runtime. The 3-step round `scratch/sm120_rounds/codex_sm120_round_libtorch_dresidual_zero_20260521` passed validation at `avg_ms=2492.696`; x10 `scratch/sm120_rounds/codex_sm120_round_libtorch_dresidual_zero_x10_20260521` passed at `avg_ms=2495.957`, slower than current native x10 `2493.133 ms`, so `dev/sm120_promotion_decisions.json` marks the row `rejected_x10_trainer_route`. |
| 2026-05-21 | LibTorch gradient-zero trainer route | Keep trainer gradient zeroing on CUDA runtime by default. The Torch C++ `cuda_memset grad_elems=124475904` row is still the fastest microbench row, and the opt-in trainer integration now exists, but the x10 TinyStories stability check regressed versus the current native selection. | Added `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `llmc/libtorch_memory.cpp`, and a cached BF16 `from_blob` handle over `model->grads_memory`; startup prints `grad_zero_backend \| Torch C++` for the opt-in build. The 3-step round `scratch/sm120_rounds/codex_sm120_round_libtorch_grad_zero_20260521` passed validation at `avg_ms=2487.730`; x10 `scratch/sm120_rounds/codex_sm120_round_libtorch_grad_zero_x10_20260521` passed at `avg_ms=2495.623`, slower than current native x10 `2493.133 ms`, so `dev/sm120_promotion_decisions.json` marks the row `rejected_x10_trainer_route`. |
| 2026-05-21 | Wide bias-add Triton closure | Keep the GPT-2 wide bias-add row on the existing trainer-callable CUDA vec2 path. The optional-stack matrix selected Triton by a tiny earlier edge, but focused same-session evidence now has source-default CUDA ahead; a wider CUDA vec4 candidate was tested and rejected. | Target-context focused A/B logs: source-default CUDA vec2 `bias_add BT=65536 OC=3072` measured `529.464 us` in `scratch/sm120_rounds/bias_add_vec2_repeats15_20260521.log`, Triton measured `530.461 us` in `scratch/sm120_rounds/bias_add_vec4_20260521_triton_runtime.log`, and the vec4 CUDA candidate measured `538.793 us` in `scratch/sm120_rounds/bias_add_vec4_default_repeats15_20260521.log`. The rejected vec4 candidate still passed `test_bias` and a 3-step TinyStories round at `avg_ms=2487.089`, but the focused runtime row did not justify keeping the source change. Current selection now marks the Triton row `rejected_same_session_refresh`. |
| 2026-05-21 | Trainer gradients-zero Torch row | Added the real trainer `grads_memory` zeroing shape to the runtime objective and kept it in the resolved Torch-family table. LibTorch C++ is the fastest measured row for `cuda_memset grad_elems=124475904`, but the integrated trainer route is rejected by x10 stability evidence. | Target-context refreshes measured CUDA runtime `148.470 us`, Python Torch `148.104 us`, and Torch C++ `147.749 us` for `grad_elems=124475904`; LibTorch parity passed in `scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521/bench_sm120_libtorch_runtime.log`. The opt-in trainer rounds above supersede the old missing-call-site gate. The optional/native source artifacts, current selection, and audit now report `trainer_rows=43`, `torch_runtime_rows=17`, `libtorch_runtime_rows=5`, `libtorch_parity_rows=5`, `project_torch_fastest_row_count=9`, and `active_promotions=0`. |
| 2026-05-21 | LibTorch trainer-link probe | Added a standalone executable link/runtime probe for LibTorch C++ memory routes. This does not promote Torch C++ memory into the trainer by itself, but it removes the vague dependency question from the next decision: future rounds can now prove whether the LibTorch route links and runs outside a Python extension before any TinyStories trainer smoke is attempted. | `dev/validate_libtorch_trainer_link.py` builds an executable that links `torch`, `torch_cuda`, `c10`, `c10_cuda`, and `cudart` without `torch_python`, wraps CUDA allocations with BF16 `from_blob` tensors, then verifies zero/copy parity. Target-context `scratch/sm120_rounds/libtorch_trainer_link_20260521/validate_libtorch_trainer_link.log` records compile PASS, runtime PASS, and probe PASS. The harness records `RUN_LIBTORCH_TRAINER_LINK_PROBE`, the manifest stores `run_libtorch_trainer_link_probe`, and the validator/audit now reject requested trainer-link probes when the log is missing or incomplete. The gradients-zero and dresidual-zero rows now have opt-in trainer call-sites plus x10 rejection evidence; the remaining logits-sized memory rows stay benchmark-only unless a real trainer call path is identified. Regenerated `current-sm120-audit.{json,md}` passes with `checks=130` and `libtorch_trainer_link_logs=1`. |
| 2026-05-21 | cuDNN packed attention saved-backward refresh | Keep the trainer on packed TK attention. The previous cuDNN packed-QKV benchmark overstated backward cost by timing a fresh cuDNN forward inside the packed backward loop; the fixed saved-forward route is much closer but still does not beat packed TK enough to justify a cuDNN trainer link. Future optional-stack rounds now carry an explicit saved-forward route marker so this benchmark contract cannot silently drift. | `dev/bench_sm120_cudnn_attention.py` now reuses the saved cuDNN forward tuple when timing packed backward and prints `cuDNNPacked Attention Backward route: saved-forward`. Target-context `scratch/sm120_rounds/cudnn_attention_saved_bwd_20260521/bench_sm120_cudnn_attention.log` measured separated cuDNN forward/backward `686.811/2371.942 us` and packed cuDNN forward/backward `807.672/2817.216 us`, for a packed total of `3624.888 us`; current packed TK evidence is about `3507-3529 us`, so no selection change. The harness records `LLMK_CUDNN_PACKED_BACKWARD_ROUTE`, the manifest stores `cudnn_packed_backward_route`, and validator/audit self-tests reject new saved-forward manifests without the marker while accepting legacy logs without that manifest field. |
| 2026-05-21 | LibTorch C++ GELU route check | Keep GPT-2 MLP GELU forward on the current CUDA trainer route. The optional-stack comparison had kept Python Torch as a tiny operator win, but comparing against the current native trainer-backed round and a focused LibTorch C++ raw-pointer route removes that promotion hypothesis. | Target-context `scratch/sm120_rounds/libtorch_runtime_gelu_20260521/bench_sm120_libtorch_runtime.log` passed LibTorch C++ parity for `gelu_forward BT=65536 C=3072` and measured `547.598 us`. The current native trainer-backed selection has CUDA `528.334 us`, and the optional Python Torch row was `529.467 us`, so `dev/write_sm120_current_selection.py` now lets a faster native row supersede an optional-stack row in `project_fastest_selection`. Regenerating `current-sm120-selection.{json,md}` now reports `project_torch_fastest_row_count=8`, `project_fastest_used_row_count=33`, and `project_fastest_resolved_divergence_row_count=9`; `current-sm120-audit.{json,md}` passes with `checks=130`, includes a native-supersession guard, verifies `1/1` supplemental LibTorch GELU timing/parity rows, and verifies the standalone trainer-link probe log. |
| 2026-05-21 | LibTorch C++ all-shape dWeight refresh | Keep all GPT-2 dWeight and accumulated dWeight rows on the current cuBLAS/cuBLASLt trainer providers. The all-shape standalone LibTorch C++ route now has structured parity/timing evidence for every GPT-2 matmul shape, but it is slower than both the trainer-backed native selection and the refreshed optional fastest rows for every tested `dW` / `dW+accum` row. | Target-context `scratch/sm120_rounds/libtorch_matmul_all_shapes_20260521/bench_sm120_libtorch_matmul.json` records LibTorch C++ parity PASS for `qkv`, `attproj`, `fc`, `fcproj`, and `lmhead` `dW` plus `dW+accum`. Timings were `1068.64/1033.98 us` for qkv, `341.89/351.59 us` for attproj, `1352.52/1366.18 us` for fc, `1376.67/1395.62 us` for fcproj, and `22204.38/22299.39 us` for lmhead. Compared with `current-sm120-selection.json`, those rows are slower than the effective native trainer rows by about `2.5%` to `6.6%`, so no Torch matmul trainer promotion is justified. `dev/audit_sm120_optimization_goal.py` now has dedicated checks for the all-shape route marker, structured evidence, `10/10` parity/timing coverage, and slower-than-trainer comparison; regenerating `current-sm120-audit.{json,md}` passes with `checks=126`. |
| 2026-05-21 | Combined fastest dInput/dGELU trainer rows | Rejected the direct composition of the fastest trainer-callable matmul microbench rows: direct cuBLAS dInput for attention projection and MLP-up plus TK fused dGELU dInput for MLP projection. This was the explicit "combine the best component rows" test, and it has a current-state edge, but it still does not beat the stable training baseline. | Candidate `codex_sm120_combo_project_fastest_dinp_tk_dgelu_x3_20260521` passed all nine focused CUDA correctness smokes and round validation, with startup confirming CUDA runtime zeroing, host scalar grad scale, and fused GELU. TinyStories x3 averaged `2498.856 ms` with normal loss `11.033154 -> 10.609922`; restored default/no-candidate x3 control averaged `2506.717 ms`. The candidate is still `5.723 ms` slower than the stable `2493.133 ms` x10 baseline, so it is recorded in `training-combinations.{md,json}` as `rejected-same-session-win`, not promoted. |
| 2026-05-21 | All-native winner composition | Rejected the broader all-native composition of fastest trainer-callable rows: direct cuBLAS dInput for attention projection and MLP-up, TK fused dGELU dInput for MLP projection, one-block-per-SM LayerNorm backward, and 1024-thread wide bias-add. It is the fastest current-state combination in this pass, but still not a stable training improvement. | Candidate `codex_sm120_combo_all_native_winners_x3_20260521` passed all nine focused CUDA correctness smokes and round validation. TinyStories x3 averaged `2498.178 ms` with normal loss `11.033154 -> 10.609906`, which is `0.678 ms` faster than the narrower dInput/dGELU combo and `8.539 ms` faster than the same-session restored default x3 control. It remains `5.045 ms` slower than the stable `2493.133 ms` x10 baseline, so it is tracked as `rejected-same-session-win` and the default/no-candidate binary was restored. |
| 2026-05-21 | LibTorch C++ dWeight result | Keep the trainer on the current cuBLAS dWeight path for the GPT-2 MLP-up row. The Python Torch row remains the fastest observed operator prototype in the older optional-stack artifact, but the standalone C++ raw-pointer route does not preserve the edge and does not justify a trainer LibTorch dependency. | Target-context `scratch/sm120_rounds/libtorch_matmul_fc_20260521/bench_sm120_libtorch_matmul.log` passed LibTorch C++ parity for `dW` and `dW+accum`, then measured `1355.76 us` and `1372.33 us`. The same current-selection evidence has optional cuBLAS at `1330.140 us` for `dW` and `1313.760 us` for `dW+accum`, while the native trainer-backed x10 round has cuBLAS at `1309.120 us` and `1333.200 us`; `dev/sm120_promotion_decisions.json` records the Torch `fc dW` row as inactive with this evidence. |
| 2026-05-21 | LibTorch C++ dWeight matmul probe | Added a standalone LibTorch C++ API benchmark for GPT-2 dWeight matmul rows. Future optional-stack rounds run it across all GPT-2 matmul shapes by default (`qkv attproj fc fcproj lmhead`) instead of only the original focused `fc` row. This does not promote Torch into the trainer yet; it creates the missing C++ raw-pointer evidence path needed to decide whether Python Torch dWeight wins survive outside Python. | `dev/bench_sm120_libtorch_matmul.py` wraps existing BF16 CUDA buffers with cached `from_blob` handles and times `dW` / `dW+accum` through C++ `mm_out` / `addmm_out`. The round harness records `RUN_LIBTORCH_MATMUL_BENCHMARKS` and `LLMK_LIBTORCH_MATMUL_SHAPES`, now defaults `LLMK_LIBTORCH_MATMUL_SHAPES` to every GPT-2 matmul shape, and the validator/audit reject future manifests with `run_libtorch_matmul_benchmarks=1` unless `bench_sm120_libtorch_matmul.log` exists, has the standalone C++ cached-`from_blob` route marker, and contains exact requested `dW` / `dW+accum` rows. Host checks passed `py_compile`, validator/audit/current-selection self-tests, and dry-run command-shape verification; the original target-context `fc` run passed parity and recorded `1355.76 us` / `1372.33 us` for `dW` / `dW+accum`. |
| 2026-05-21 | LibTorch round route control | Wired the standalone LibTorch C++ API runtime route into `scripts/run_sm120_optimization_round.sh` as the default future LibTorch memory benchmark route. This keeps current selection evidence compatible with older logs while making the next optional-stack refresh test the linked-C++ route shape by default. | The harness now records `LLMK_LIBTORCH_RUNTIME_ROUTE`, passes it to `dev/write_sm120_round_manifest.py`, and invokes `dev/bench_sm120_libtorch_runtime.py --route "$LLMK_LIBTORCH_RUNTIME_ROUTE"`. Validator and audit self-tests now enforce that new manifest `libtorch_runtime_route` values match the raw-pointer marker recorded in `bench_sm120_libtorch_runtime.log`, while older artifacts without a recorded route may still use either raw-pointer marker. |
| 2026-05-21 | LibTorch C++ API trainer-link probe | Added `--route cxx-api-raw-pointer` to `dev/bench_sm120_libtorch_runtime.py`. This does not promote a Torch memory route by itself; it creates the missing linked-C++ evidence path for the two Torch-fast memory situations by timing cached `from_blob` handles through a standalone LibTorch shared library rather than a Python extension. | Host-side `py_compile` passed, `--help` exposes `cxx-api-raw-pointer`, and validators now accept either the existing extension-backed raw-pointer marker or the standalone C++ API marker as LibTorch raw-pointer route evidence. The target-context raw-pointer refresh records the actual linked-C++ parity/timing evidence used for current selection decisions. |
| 2026-05-21 | LibTorch C++ raw-pointer runtime route | Updated `dev/bench_sm120_libtorch_runtime.py` so the default LibTorch C++ route wraps existing CUDA pointers with cached `from_blob` tensors before timing zero/copy. Keep the trainer on CUDA runtime memory operations for now: the raw-pointer C++ route is now closer to trainer-callable for zeroing, while copy rows are explicitly profiler-runtime benchmark evidence rather than trainer replacement rows. | Target-context raw-pointer LibTorch passed full-row parity and measured hidden/logits memset `60.011/3984.000 us` and copy `131.526/8686.496 us`. Same-session native CUDA memory retuning with non-streaming 1024-thread kernels measured hidden/logits memset `60.056/4016.422 us` and copy `133.416/8881.951 us`, so the native replacement does not beat the Torch/CUDA choices. Revalidating the optional round passed with `benchmarks=194`, `torch_objective_rows=42`, `libtorch_runtime_rows=4`, `libtorch_parity_rows=4`, and `active_promotion_candidates=0`; regenerating current selection/audit now reports `optional_non_trainer=12`, `torch_selected=9`, `checks=115`, and `active_promotions=0`, including required `LibTorch runtime raw-pointer route`, `runtime copy call-path kind`, fastest-row debt-count checks, `6/6` trainer/C++ callable debt rows with trainer-smoke, x10, or stability evidence, a Torch fastest-row partition of `0` trainer-used, `6` resolved, and `3` extra/non-objective rows, `9/9` Torch disposition rows with action/reason, plus `14/14` resolved project-fastest rows linked to decision rows and `8/8` non-trainer resolved rows carrying action metadata. |
| 2026-05-21 | LibTorch C++ runtime memory probe | Added `dev/bench_sm120_libtorch_runtime.py` to the optional Python-stack benchmark phase and validator contract for the exact memory rows where Torch remains faster as an operator prototype. Keep the trainer on CUDA runtime memory operations for now: the C++ Torch API path is feasible and selected for two project-wide fastest memory rows after full-row parity checks, but there is still no trainer dependency change or TinyStories smoke gate for libtorch integration. | The probe builds a tiny C++ extension, falls back to direct `c++` when `ninja` is missing, and verifies full-row zero/copy parity before timing. Same-session target-context `LLMK_BENCH_REPEATS=9 ./bench_sm120_runtime` measured CUDA runtime hidden/logits memset `59.647/3952.352 us` and copy `131.799/8769.241 us`; Python Torch measured `59.814/3928.672 us` memset and `131.677/8666.176 us` copy. The current optional-stack `bench_sm120_libtorch_runtime.log` records parity PASS for hidden/logits zero/copy and measured Torch C++ `59.850/3917.600 us` memset plus `131.987/8662.080 us` copy. Revalidating the optional round now requires all four LibTorch timing and parity rows and passed with `benchmarks=194`, `torch_objective_rows=42`, `libtorch_runtime_rows=4`, `libtorch_parity_rows=4`, and `active_promotion_candidates=0`; regenerating current selection/audit reports `optional_non_trainer=11`, `torch_selected=8`, `checks=106`, and `active_promotions=0`, with every resolved optional decision carrying explicit `decision_evidence`, every project-wide fastest divergence carrying a rendered reason/evidence summary, all `29` project-wide fastest rows used by the trainer listed explicitly, the persisted used/resolved/extra partition identities and row contents audited, every selected row carrying manifest-checked build/runtime flags plus matching source run label, artifact directory, and git commit, every source selected-backends row matching its generated `Selected Backend Rows` scoreboard row, every source selected-backend artifact carrying valid schema/run/policy/count/Torch/LibTorch coverage metadata, every project-wide fastest/Torch-fastest row content checked against the source optional `selected-backends.json`, every native trainer row checked against the native source selected row or rejected-winner fallback mapping, every native extra benchmark-only row checked against the source non-objective selected rows, every resolved optional decision checked against its source promotion-candidate or selected-backend row, and every native/optional attention route total checked against its source `attention_route_rows`. |
| 2026-05-21 | SM120 bias-add block-size hook | Added behavior-preserving `LLMK_SM120_BIAS_ADD_BLOCK_SIZE` and `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE` hooks for future focused A/B work, but kept the default launch block unchanged. A one-off `1024` candidate row improved `OC=3072`, but the source-default shape-aware rerun did not reproduce a material win, and `512` was noise-level. | Default rebuild passed `test_bias` and measured `bias_add OC=768/3072` at `80.698/536.701 us` with `LLMK_BENCH_REPEATS=9`. Candidate `512` passed the same smoke and measured `81.521/535.581 us`; candidate `1024` passed and measured `81.111/527.753 us`; source-default shape-aware rerun measured `80.931/535.270 us`; final restored-default rebuild passed `test_bias` and measured `81.034/536.434 us`. |
| 2026-05-21 | cuBLASLt MLP epilogue algos  | Added a focused cuBLASLt heuristic-index probe for the trainer-shaped fused MLP epilogue rows. Keep the default lowest-waves cuBLASLt selection: the returned per-algorithm candidates do not beat the current default for either `fc` forward+GeLU or `fcproj` dInput+dGeLU, so there is no route-specific trainer promotion. | New `dev/cuda/bench_sm120_cublaslt_epilogue_algos.cu` target built with `make -B -j 4 bench_sm120_cublaslt_epilogue_algos DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`. Target-context `LLMK_BENCH_REPEATS=7 ./bench_sm120_cublaslt_epilogue_algos` on RTX 5090 returned two algorithms for each row: `fc fwd+GeLU` default idx `0` was best at `1437.429 us` versus idx `1` `1449.168 us`; `fcproj dInp+dGeLU` default idx `0` was best at `1760.613 us` versus idx `1` `1789.392 us`. |
| 2026-05-21 | Torch attention layout route | Added a focused Torch route probe for the unresolved separated-Q/K/V attention opportunity. Keep the trainer on the current qkv projection plus packed TK attention route: split-strided Torch improves the backward-side route versus packed Torch, but the layout rewrite does not beat the current combined qkv+attention route once qkv projection and bias-gradient work are included. | New `dev/bench_sm120_torch_attention_layouts.py` target-context run on RTX 5090 with `--repeats 5 --warmup 2 --json-out scratch/sm120_rounds/torch_attention_layout_probe_20260521.json` measured `TorchQKVSinglePacked` `2030.240/6334.176 us`, `TorchQKVSplitStrided` `2108.192/4822.080 us`, and `TorchQKVSplitMaterialized` `2653.696/5388.320 us` for forward/backward. The current selected native qkv+attention route is `1824.291 us` forward (`qkv fwd 1039.600 + TK attention fwd 784.691`) and about `4952.834 us` backward including qkv dInput, qkv dWeight, and qkv bias-gradient rows, for about `6777.125 us` combined versus `6930.272 us` for the best Torch split-strided route. |
| 2026-05-21 | Attention dprep warp retest | Rechecked the older `LLMK_SM120_DPREP_WARPS=3` attention-prep setting against the current stream-sync default because historical notes mentioned it as a default while the current source uses `4`. Keep the source default at `4`: the retest is correctness-clean but the timing edge is noise-level and does not justify a trainer smoke or default change. | Default rebuild `make -B -j 4 test_attention bench_sm120_attention train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` passed, target-context `./test_attention` passed, and `LLMK_BENCH_REPEATS=9 ./bench_sm120_attention` measured `781.911 us` forward / `2742.730 us` backward. Candidate rebuild with `EXTRA_NVCC_FLAGS=-DLLMK_SM120_DPREP_WARPS=3` passed the same smoke and measured `783.235 us` forward / `2741.640 us` backward. |
| 2026-05-21 | SM120 backward stream sync | Promoted the SM120 trainer's final backward synchronization from a device-wide sync to a `main_stream` sync, with `LLMK_SM120_DISABLE_BACKWARD_STREAM_SYNC` as the opt-out. This is trainer-callable source behavior rather than an optional-stack reference row: all auxiliary split-K work is joined back to `main_stream`, and the host only needs the queued loss copy before update timing continues. | Opt-in `codex_sm120_round_backward_stream_sync_20260521` passed all native gates at `avg_ms=2483.598` over 3 steps, and opt-in x10 `codex_sm120_round_backward_stream_sync_x10_20260521` validated at `2495.290 ms`. The promoted default-build x10 round `codex_sm120_round_backward_stream_sync_default_x10_20260521` passed correctness, manifest, stack-probe, benchmarks, checkpoint cleanup, and TinyStories training with `avg_ms=2493.133`, ahead of the prior stable x10 `2495.443`, with loss `11.032358 -> 9.588612`, norm `22.1414 -> 6.3099`, `use_master_weights disabled`, and `gelu_fusion 1`. The regenerated current selection/audit now uses this native round, reports `native_selected_row_count=42`, `native_inactive_selected_row_count=5`, `project_fastest_resolved_divergence_row_count=13`, `project_torch_fastest_row_count=8`, and passes with `checks=88`, `active_promotions=0`. |
| 2026-05-21 | Attention route totals | Added route-level attention totals to the validator and consolidated current selection, so trainer-shaped routes are compared as forward+backward units instead of only as split selected rows. The current evidence keeps packed TK as the trainer route: native TK totals `787.769 + 2741.750 = 3529.519 us`; optional-stack packed TK is `3507.412 us`, cuDNNPacked is `4317.172 us`, TorchPacked is `5135.992 us`, TorchMaterializedPacked is `5456.421 us`, and TritonPacked remains forward-only because backward is not implemented. Separated Torch remains the fastest reference route at `2751.159 us`, but it is not trainer-layout-compatible. | Revalidated `codex_sm120_round_current_native_x10_median_20260521` and `codex_sm120_round_torch_attention_materialized_20260521` with `dev/validate_sm120_round.py`, regenerating `scoreboard-candidates.md`, `selected-backends.json`, and `promotion-candidates.json` with `Attention Route Totals` / `attention_route_rows`. Regenerating `current-sm120-selection.{json,md}` and `current-sm120-audit.{json,md}` now passes with `checks=88`, `native_attention_route_rows=1`, `optional_attention_route_rows=8`, and the audit check `attention route totals` passing. |
| 2026-05-21 | Current-selection promotion metadata | Preserved the actionable promotion queue fields in `current-sm120-selection`. Resolved optional-stack decisions now carry `candidate_class`, `priority`, `promotion_gate`, decision notes, and measured edge fields from `promotion-candidates.json`; the Markdown table shows class and gate so the consolidated artifact can be used directly as the next-work queue. | `python3 dev/write_sm120_current_selection.py --self-test` and `python3 dev/audit_sm120_optimization_goal.py --self-test` passed. Regenerating `current-sm120-selection.{json,md}` and `current-sm120-audit.{json,md}` now passes with `checks=88`; the audit includes `resolved optional promotion metadata` with `missing=0`, while preserving `project_torch_fastest_rows=8`, `triton_matmul_rows=21`, and `active_promotion_candidate_count=0`. |
| 2026-05-21 | Current-selection replay defaults | Centralized the current native and optional-stack selection-round defaults in `dev/sm120_objective_contract.py`. The selection writer and audit import those paths, and the round harness now omits selection-round CLI overrides unless `SM120_SELECTION_NATIVE_ROUND` or `SM120_SELECTION_OPTIONAL_ROUND` is explicitly set. This prevents one-command replay from drifting back to an older native round while keeping fresh-round overrides available. | `RUN_LABEL=codex_sm120_round_dry_default_selection_audit RUN_CORRECTNESS=0 RUN_BENCHMARKS=0 RUN_TRAINING=0 RUN_STACK_PROBE=0 RUN_ARTIFACT_VALIDATOR=0 RUN_CURRENT_SELECTION_AUDIT=1 DRY_RUN=1 scripts/run_sm120_optimization_round.sh` prints `write_sm120_current_selection.py --json-out ... --markdown-out ...` and `audit_sm120_optimization_goal.py --selection-json ... --selection-md ... --json-out ... --markdown-out ...` without `--native-round` / `--optional-round`. Default regeneration still passes with `checks=88`, `project_torch_fastest_rows=8`, `triton_matmul_rows=21`, and `active_promotion_candidate_count=0`. |
| 2026-05-21 | Current native x10 median round | Refreshed the native current-source x10 evidence with the median attention benchmark and the current 93-row native benchmark contract. The trainer mix remains the native C++/CUDA/TK/cuBLAS/cuBLASLt stack; the run is valid current evidence but not a new stable-best claim versus the earlier x10 `2495.443 ms` row. | Target-context `RUN_LABEL=codex_sm120_round_current_native_x10_median_20260521 MAX_STEPS=10 RUN_PYTHON_STACK_BENCHMARKS=0 RUN_CURRENT_SELECTION_AUDIT=1 LLMK_BENCH_REPEATS=7 scripts/run_sm120_optimization_round.sh` passed correctness, manifest, stack-probe, benchmark, training, current-selection, and audit gates with `benchmarks=93`, `family_stack_rows=168`, `train_steps=10`, `avg_ms=2496.245`, attention `787.769/2741.750 us`, and `active_promotions=0`. |
| 2026-05-21 | Trainer-row exact filter | Tightened `dev/write_sm120_current_selection.py` so the published native trainer mix contains exactly the GPT-2 objective rows from `dev/sm120_objective_contract.py`. Extra native benchmark rows, such as LayerNorm `C=3072` stress timings, remain visible as benchmark evidence but no longer inflate `trainer_rows`. | Regenerating current selection from `codex_sm120_round_current_native_x10_median_20260521` now reports `native_source_selected_row_count=45`, `native_selected_row_count=42`, and `native_extra_selected_row_count=3`; the three extra rows are LayerNorm `N=65536 C=3072` forward/fused/backward. The project audit now requires exactly `42` trainer rows and passes with `checks=64`. |
| 2026-05-21 | Project-wide Torch fastest rows | Made the consolidated current-selection artifact compare optional-stack winners against the current native trainer-backed row before reporting the project-wide fastest set. Torch wins are still listed directly by exact suite/kernel/shape when they actually beat the current native evidence, but stale optional wins are superseded by native rows. | Regenerating `current-sm120-selection.{json,md}` now reports `project_fastest_row_count=49`, `project_torch_fastest_row_count=8`, `project_trainer_callable_row_count=38`, `project_fastest_used_row_count=33`, `project_fastest_resolved_divergence_row_count=9`, `project_fastest_extra_row_count=7`, and `project_fastest_unresolved_objective_row_count=0`. The Python Torch GELU-forward row is superseded by the current native CUDA row (`528.334 us` versus `529.467 us`), and the focused LibTorch C++ GELU probe measured `547.598 us`, so GELU forward stays on CUDA. The project audit now verifies that an optional row cannot remain in `project_fastest_selection` when the native trainer-backed row is faster; it passes with `checks=130`, `trainer_rows=42`, `project_torch_fastest_rows=8`, `torch_objective_rows=42`, `cutedsl_gemm_rows=5`, `triton_matmul_rows=21`, `cudnn_attention_rows=4`, `triton_attention_unavailable=2`, `torch_runtime_rows=16`, `libtorch_runtime_rows=4`, `libtorch_supplemental_runtime_rows=1`, `libtorch_trainer_link_logs=1`, `triton_runtime_unavailable=10`, and `active_promotions=0`. |
| 2026-05-21 | fcproj dInput benchmark-context flip | Recorded the optional-stack `fcproj` dInput cuBLASLt selection as an inactive benchmark-context flip, then resolved the older current-source conflict with newer stream-sync trainer evidence. The prior x10 native round chose cuBLAS for this row, but the promoted stream-sync x10 round and optional-stack benchmark both choose cuBLASLt, so the trainer mix keeps cuBLASLt here. | Revalidating `codex_sm120_round_torch_attention_materialized_20260521` annotates the row as `benchmark_context_flip`: optional stack `cuBLASLt 1372.020 us` versus `cuBLAS 1415.140 us`. The older current-source x10 round chose `cuBLAS 1380.080 us` versus `cuBLASLt 1405.130 us`, but `codex_sm120_round_backward_stream_sync_default_x10_20260521` selects `cuBLASLt 1381.460 us` versus `cuBLAS 1383.670 us` and validates with `avg_ms=2493.133`. |
| 2026-05-21 | Current-selection training guard | Tightened `dev/write_sm120_current_selection.py` so the native round used for the published trainer mix must include TinyStories training evidence. Benchmark-only native rounds can still be inspected with an explicit override, but they no longer pass as current trainer-mix artifacts by default. | `python3 dev/write_sm120_current_selection.py --self-test` covers the benchmark-only rejection and explicit bypass. Default generation with the stable x10 native round passed. Using `codex_sm120_round_native_attention_median_20260521` without the override now fails because its manifest records `run_training='0'`; the same command with `--allow-benchmark-only-native` succeeds only as an inspection artifact. |
| 2026-05-21 | qkv dInput microbench flip | Recorded the fresh native qkv dInput cuBLAS selection as an inactive noise-floor flip, not a trainer promotion. The benchmark-only round selected cuBLAS over cuBLASLt by about 0.2%, but the stable x10 selection artifact has cuBLASLt ahead and there is no trainer-smoke evidence for changing this row. | `codex_sm120_round_native_attention_median_20260521` selected qkv dInput cuBLAS `1012.36 us` versus cuBLASLt `1014.27 us`; current stable x10 selection records cuBLASLt `1007.09 us`. `dev/sm120_promotion_decisions.json` marks the cuBLAS row `noise_floor_microbench_flip`, and revalidating the round annotates `selected-backends.json` plus `scoreboard-candidates.md` with that inactive decision. |
| 2026-05-21 | Attention benchmark repeatability | Upgraded `bench_sm120_attention` to the same median repeated-event timing contract as the other SM120 native microbenchmarks. This is an evidence-quality change only: the current trainer route remains packed-QKV TK, while faster Torch SDPA remains scoped to already-separated Q/K/V reference/layout-rewrite rows because packed and materialized-packed Torch rows are slower for the trainer layout. | `make -B -j 4 bench_sm120_attention DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` passed. Target-context `LLMK_BENCH_REPEATS=7 ./bench_sm120_attention` reported packed TK attention forward/backward `783.509/2739.042 us`. The benchmark-only round `codex_sm120_round_native_attention_median_20260521` then passed build, stack-probe, manifest, all nine correctness smokes, benchmark validation, and selected-backend generation with `benchmarks=93`, selected attention rows `785.867/2746.680 us`, and `0` promotion candidates. |
| 2026-05-21 | Python-stack GPU wording | Removed remaining optional-benchmark messages that could be read as GPU availability claims. A missing `torch.cuda` context is now reported as a process-local PyTorch CUDA context issue, while backend-specific exact-shape rejection rows remain scoped to that backend path. | Updated the Torch, Triton, and cuDNN Python benchmark scripts plus the optional-stack summary grep patterns. Follow-up verification used `python3 -m py_compile` across the changed scripts, `bash -n scripts/run_sm120_optimization_round.sh`, `git diff --check`, and a stale GPU wording scan. |
| 2026-05-21 | SM120 atomic-dQ attention trainer check | Rejected `LLMK_SM120_ATOMIC_DQ` for GPT-2 training. This was the remaining trainer-shaped attention alternative, but it disables the SM120 packed-QKV fast path and regresses badly, so packed TK attention remains the trainer route. | Added an atomic-dQ branch to `dev/cuda/bench_sm120_attention.cu` so the benchmark compiles against the generic `attention_forward`/`attention_backward` route when `LLMK_SM120_ATOMIC_DQ` is set. Target-context round `scratch/sm120_rounds/codex_sm120_attention_atomic_dq_x3_20260521` passed all nine CUDA correctness smokes, with `test_attention` printing `SM120 packed-QKV fast path: no`, then TinyStories x3 regressed to `avg_ms=2945.101`. Restored default/no-candidate binaries and rechecked default `bench_sm120_attention` at `786.763/2741.636 us` forward/backward. |
| 2026-05-21 | Classifier exp2 softmax math | Rejected the opt-in SM120 classifier `exp2f(x * log2(e))` math path as a default. The candidate keeps a compile-time A/B hook, passes the classifier smoke, and does not affect the default build, but same-session focused timing favored the current `expf` path for both loss-only and training dlogits rows. | Candidate build with `EXTRA_NVCC_FLAGS=-DLLMK_SM120_CLASSIFIER_EXP2` passed `./test_fused_classifier` and measured `fused_classifier_loss 3997.862 us`, `fused_classifier 8921.062 us` with `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime`. Restored default build passed `./test_fused_classifier` and measured `3943.962 us` / `8909.920 us` in the same repeat-count benchmark. |
| 2026-05-21 | Torch objective coverage guard | Made Torch benchmark coverage an exact-row contract for the whole SM120 optimization project. Python-stack rounds now fail unless Torch has parsed timing rows for all 42 GPT-2 objective rows, and the project-wide audit requires the optional-stack scoreboard's `Torch Objective Benchmark Coverage` section before reporting the current trainer mix. Torch still only enters the trainer when the winning row has a trainer-callable route and smoke evidence; otherwise it stays as resolved operator/reference evidence. | `python3 dev/validate_sm120_round.py --self-test`, `python3 dev/audit_sm120_optimization_goal.py --self-test`, and `python3 -m py_compile dev/validate_sm120_round.py dev/audit_sm120_optimization_goal.py` passed. Revalidating `codex_sm120_round_torch_attention_materialized_20260521` passed with `benchmarks=190`, `torch_objective_rows=42`, `stacks=9`, and `family_stack_rows=168`; regenerating `current-sm120-audit.{json,md}` passed with `63` checks, `42` trainer rows, `8` Torch-selected optional rows, and `42` Torch objective benchmark rows. |
| 2026-05-21 | Torch materialized packed attention | Added a materialized packed-QKV Torch SDPA attention benchmark row and made the current optional-stack selection/audit use the refreshed round. Native Torch SDPA remains faster for already-separated Q/K/V experiments, but neither the strided packed-QKV path nor explicit Q/K/V materialization beats the trainer's packed TK route, so no libtorch trainer link is promoted. The tiny Triton `bias_add OC=3072` row that appeared in the refresh is resolved as operator evidence only. | Round `codex_sm120_round_torch_attention_materialized_20260521` validated with `benchmarks=190`, `stacks=9`, `family_stack_rows=168`, and `train_steps=0`. Same round measured attention forward/backward: packed TK `778.909/2728.503 us`, native Torch separated `555.498/2195.661 us`, TorchPacked `1068.491/4067.501 us`, and TorchMaterializedPacked `1260.536/4195.885 us`. Regenerating `current-sm120-selection.{json,md}` passed with `optional_non_trainer=11`, `optional_decisions=14`, and `0` active promotions; `current-sm120-audit.{json,md}` passed with `61` checks, `42` trainer rows, `8` Torch-selected optional rows, and `0` active promotions. |
| 2026-05-21 | TK exact dGELU selector | Reopened the SM120 TK fused dGELU dInput near-miss with an opt-in exact-derivative selector. The route now passes the trainer-shaped matmul smoke and keeps a focused `dInp+dGeLU` row edge, but it is rejected as the default because the x10 TinyStories stability round regressed versus the current stable x10 backend mix. | `EXTRA_NVCC_FLAGS=-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP ./test_matmul` passed `10/10`, including GPT-2 fcproj dGELU max diff `0.125000`. Focused `LLMK_BENCH_REPEATS=7 ./bench_sm120_matmul` measured TK exact `1761.68 us` versus cuBLASLt fused `1855.33 us`. Candidate round `codex_sm120_round_tk_fused_dgelu_exact_20260521` validated at `avg_ms=2492.018`; x10 `codex_sm120_round_tk_fused_dgelu_exact_x10_20260521` regressed to `avg_ms=2502.328` versus current stable x10 `2495.443`. |
| 2026-05-21 | Round-manifest evidence audit | Extended the project-wide audit to validate both source round manifests. The audit now checks SM120 artifact config, git short commit/status path, nvcc metadata, every expected smoke/benchmark/trainer binary row from `EXPECTED_MANIFEST_BINARIES`, SHA256 evidence, and `run_python_stack_benchmarks=1` for the optional-stack round. | `python3 dev/audit_sm120_optimization_goal.py --self-test` now builds synthetic manifests with expected binaries and SHA evidence. Regenerating `current-sm120-audit.{json,md}` passed with `61` checks and `14` manifest binary rows in each source round. |
| 2026-05-21 | Stack-probe evidence audit | Extended the project-wide audit to verify the full backend-stack probe contract for both current source rounds. The audit now checks every objective stack plus GPU runtime for status, evidence, candidate-use, and next-action fields, and checks every family/stack matrix row for status, reason, and next action. | `python3 dev/audit_sm120_optimization_goal.py --self-test` now builds synthetic stack probes with the full schema. Regenerating `current-sm120-audit.{json,md}` passed with `50` checks, `9` stack-probe rows per source round, `168` family/stack rows per source round, and `0` active promotions. |
| 2026-05-21 | Full Python-stack log audit | Replaced the audit's Torch-only optional-log check with the shared `PYTHON_STACK_BENCHMARK_LOGS` contract. The project-wide audit now requires every optional Python-stack log, covering Torch, Torch C++ runtime memory, Triton, cuDNN, CuTeDSL, and the combined LayerNorm comparison log, to exist and be non-empty in the optional-stack round. | `python3 dev/audit_sm120_optimization_goal.py --self-test` includes a missing non-Torch log failure. The latest regenerated `current-sm120-audit.{json,md}` passes with `89` checks, `12` Python-stack benchmark logs, `12` optional non-trainer selected rows, `9` Torch-selected optional rows, and `0` active promotions. |
| 2026-05-21 | Current-selection optional decision table | Extended `current-sm120-selection.md` with a `Resolved Optional-Stack Decisions` table listing each optional decision by suite, kernel, exact shape, selected stack, timing, scope, and decision status. This makes the "use Torch where faster, otherwise record why not" evidence directly reviewable from the project-wide artifact instead of only as JSON. The audit now requires that table. | `python3 dev/write_sm120_current_selection.py --self-test` and `python3 dev/audit_sm120_optimization_goal.py --self-test` passed. Regenerating `current-sm120-selection.{json,md}` and `current-sm120-audit.{json,md}` passed with `38` audit checks, `17` optional non-trainer selected rows, `21` optional decisions, and `0` active promotions. |
| 2026-05-21 | Current-selection optional guard | Moved the non-trainer optional-stack decision contract into `dev/write_sm120_current_selection.py` so invalid current-selection artifacts fail during generation, not only during the later audit. The generator now records optional non-trainer selected and promotion-row counts, requires every non-trainer optional selected row to have a matching inactive promotion decision, and the audit verifies those counts against the optional round. | `python3 dev/write_sm120_current_selection.py --self-test` covers missing promotion-row and missing inactive-decision failures. Regenerating `current-sm120-selection.{json,md}` passed with `42` native rows, `17` optional non-trainer rows, `21` optional decisions, and `0` active promotions; regenerating `current-sm120-audit.{json,md}` passed with `37` checks. |
| 2026-05-21 | Optional-stack decision audit | Broadened the current-selection audit from Torch-only optional rows to all selected optional-stack rows that lack a trainer call path. Each non-trainer optional win now needs a matching `promotion-candidates.json` row, inactive decision coverage, and a consolidated entry in `current-sm120-selection.json`; Torch keeps separate benchmark-log presence checks. | `python3 dev/audit_sm120_optimization_goal.py --self-test` now includes a non-Torch optional-row negative case. Regenerating `scratch/sm120_rounds/current-sm120-audit.{json,md}` passed with `35` checks, `42` trainer rows, `17` optional non-trainer selected rows, `13` Torch-selected optional rows, and `0` active promotions. |
| 2026-05-21 | Shared objective row contract | Moved exact GPT-2 shape/pass selection requirements into `dev/sm120_objective_contract.py` so the round validator and current-selection audit use the same source of truth. This removes duplicate tool-local definitions for the GEMM pass/shape matrix and fixed runtime/operator shapes. | `python3 -m py_compile dev/sm120_objective_contract.py dev/validate_sm120_round.py dev/audit_sm120_optimization_goal.py`, `python3 dev/validate_sm120_round.py --self-test`, and `python3 dev/audit_sm120_optimization_goal.py --self-test` passed. Revalidating both source rounds and regenerating current selection/audit artifacts still passed with exact-row coverage `42/42`. |
| 2026-05-21 | Trainer exact-row coverage audit | Extended the current-selection audit from objective-family coverage to exact GPT-2 selected-row coverage. The audit now requires the native trainer mix to include every GEMM pass/shape row, attention forward/backward, LayerNorm forward/fused/backward, classifier loss and dlogits, AdamW, global norm, encoder, bias/reduction rows, and required memset/copy rows. | `python3 dev/audit_sm120_optimization_goal.py --self-test` passed with a synthetic exact-row fixture. Regenerating `scratch/sm120_rounds/current-sm120-audit.{json,md}` passed with `33` checks, `42` trainer rows, `13` Torch-selected optional rows, `0` active promotions, and trainer exact-row coverage `42/42`. |
| 2026-05-21 | Trainer family coverage audit | Extended the current-selection audit so the reported native trainer mix must cover every objective family from `dev/sm120_objective_contract.py`. This prevents a current backend-selection artifact from passing if it drops a GPT-2 operator family even when individual selected rows have valid provenance. | `python3 dev/audit_sm120_optimization_goal.py --self-test` now builds a synthetic full-family fixture. Regenerating `scratch/sm120_rounds/current-sm120-audit.{json,md}` passed with `32` checks, `42` trainer rows, `13` Torch-selected optional rows, `0` active promotions, and trainer objective-family coverage `21/21`. |
| 2026-05-21 | Provenance path audit | Strengthened `dev/audit_sm120_optimization_goal.py` so row-level provenance must point to existing files, not only non-empty strings. The audit now checks timing log, round manifest, backend-stack probe, and correctness log paths for every native trainer row and resolved optional-stack decision; runtime memcpy/memset rows carry an explicit correctness-note field because they are CUDA/Torch runtime primitives rather than route-specific kernels. | Revalidating both source rounds and regenerating `current-sm120-selection.json`/`current-sm120-audit.md` passed with `31` checks. A direct artifact scan found `63` audited rows, `0` missing referenced files, and `8` runtime primitive rows with explicit `correctness_evidence_note`. |
| 2026-05-21 | Selection row provenance | Added row-level evidence pointers to generated selected-backend and promotion artifacts. Each row now carries the source run label, artifact directory, git commit, timing log path, round-manifest path, backend stack-probe path, and relevant correctness log paths, so Torch/native decisions can be audited without manually inferring provenance from the enclosing artifact. | Revalidating the stable native x10 round and Torch-stack refresh round regenerated `selected-backends.json` and `promotion-candidates.json` with provenance fields. Regenerating `current-sm120-selection.json` preserved those fields, and `dev/audit_sm120_optimization_goal.py` now passes `31` checks including `selection row provenance` with `missing=0`. |
| 2026-05-21 | Python-stack log enforcement | Made `dev/validate_sm120_round.py` enforce the optional Python-stack benchmark log set whenever `round-manifest.json` records `run_python_stack_benchmarks=1`. This closes a coverage hole where Torch/Triton/cuDNN/CuTeDSL logs were parsed if present but could be omitted without failing the round. | `python3 dev/validate_sm120_round.py --self-test` now includes a missing-Torch-log negative case. Revalidating `scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521` with manifest, stack-probe, correctness, and benchmark gates passed with `benchmarks=187`, `stacks=9`, `family_stack_rows=168`, and the scoreboard now includes `Python Stack Benchmark Logs` rows for all optional logs including `bench_sm120_torch_matmul.log` and `bench_sm120_torch_runtime.log`. |
| 2026-05-21 | SM120 selection harness hook | Added an opt-in `RUN_CURRENT_SELECTION_AUDIT=1` path to `scripts/run_sm120_optimization_round.sh`. The harness can now regenerate `current-sm120-selection.{json,md}` and run `dev/audit_sm120_optimization_goal.py` after round validation, with overrideable native/optional round paths for fresh evidence. | `bash -n scripts/run_sm120_optimization_round.sh` passed. `python3 dev/audit_sm120_optimization_goal.py --self-test` now includes a negative check that fails when a resolved Torch-selected optional row is missing from the consolidated selection. The refreshed audit passed with `30` checks, `42` trainer rows, `13` Torch-selected optional rows, and `0` active promotions. |
| 2026-05-21 | SM120 optimization goal audit | Added a host-side audit for the mixed-backend goal. It verifies the current selection artifact, native x10 training evidence, optional-stack matrix coverage, Torch benchmark logs, and the rule that faster Torch rows must be either trainer-callable or explicitly resolved as inactive/reference rows before the trainer mix is reported. | `python3 dev/audit_sm120_optimization_goal.py --self-test` passed. `python3 dev/audit_sm120_optimization_goal.py --json-out scratch/sm120_rounds/current-sm120-audit.json --markdown-out scratch/sm120_rounds/current-sm120-audit.md` passed with `29` checks, `42` trainer rows, `13` Torch-selected optional rows, `0` active promotion candidates, `168` family/stack rows in both audited rounds, and native x10 `avg_ms=2495.443`. |
| 2026-05-21 | Consolidated SM120 selection | Added a generated current-selection artifact that joins the stable native x10 trainer mix with the latest Torch/Triton/cuDNN/CuTeDSL comparison evidence. It applies inactive decision-registry rows before reporting the effective trainer stack, so the rejected attproj direct-cuBLAS dInput and LM-head direct-cuBLAS forward microbench wins map back to cuBLASLt in the current trainer mix. Promotion-only decisions are also merged so non-equivalent Torch reference wins, such as BF16-state AdamW, cannot disappear from the resolved-decision summary. | `python3 dev/write_sm120_current_selection.py --self-test` covers active optional promotions, non-trainer-callable native rows, inactive native rows without a fallback, and promotion-only resolved decisions. `python3 dev/write_sm120_current_selection.py --json-out scratch/sm120_rounds/current-sm120-selection.json --markdown-out scratch/sm120_rounds/current-sm120-selection.md` passed with `native_rows=42`, `inactive_native=2`, `optional_decisions=21`, and `active_promotions=0`. Effective stack counts are CUDA `15`, CUDA runtime `4`, TK packed-QKV `2`, cuBLAS `10`, cuBLASLt `10`, and cuBLASLt fused `1`. |
| 2026-05-21 | Stable x10 artifact refresh | Refreshed the current stable x10 default round's stack probe and derived selected-backend artifacts to the current optimisation contract without rerunning training. The run remains the stable same-harness baseline: all selected rows already have C++/CUDA/TK trainer routes, so there are no active promotion candidates in that native-only round. | `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520/backend-stacks.md` recorded Torch/Triton/cuDNN/CuTeDSL as available. Revalidating the existing round with `--require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints` passed with `benchmarks=86`, `stacks=9`, `family_stack_rows=168`, `train_steps=10`, and `avg_ms=2495.443`; `selected-backends.json` has `42` selected rows and `promotion-candidates.json` has `0` active / `0` total. |
| 2026-05-21 | GPU-runtime probe wording | Removed stale raw NVML failure text from the stack-probe source path and generated scratch artifacts. Failed metadata queries are now recorded as process-context metadata misses, while target runtime is proven by explicit SM120 correctness and benchmark logs. The project audit now rejects stale GPU runtime wording and raw NVML metadata-failure phrases so this cannot drift back into the goal docs or stack-probe source. | `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out /tmp/sm120_backend_stacks_sanitized.json --markdown-out /tmp/sm120_backend_stacks_sanitized.md` passed and recorded Torch `2.11.0+cu130`, Triton `3.6.0`, cuDNN `9.22.0`, CuTeDSL `4.5.1`, and GPU runtime `available` without raw NVML failure text. `python dev/audit_sm120_optimization_goal.py --self-test` now covers stale GPU wording, and regenerating `scratch/sm120_rounds/current-sm120-audit.{json,md}` passes with `checks=132`, including `GPU runtime wording docs`. A repo-wide wording scan including `scratch/` found no stale GPU availability phrases. |
| 2026-05-21 | Selected-backend decision annotations | Extended the SM120 validator so registry decisions also attach to selected rows that already have a C++/CUDA benchmark route. This prevents raw fastest-row artifacts from being mistaken for promoted trainer defaults when a route was explicitly rejected by correctness or TinyStories evidence. | `dev/validate_sm120_round.py --self-test` passed. Revalidating `scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521` passed with `benchmarks=187`, `stacks=9`, and `family_stack_rows=168`; `selected-backends.json` now has `19` rows with `decision_status`, including rejected attproj TK forward, attproj direct-cuBLAS dInput, TK fused dGELU dInput, and LM-head direct-cuBLAS forward. |
| 2026-05-21 | Torch stack refresh | Refreshed the full Python/Torch stack benchmark matrix against the current C++ source. Torch remains selected where it is actually fastest for Python/operator scope, but no new Torch trainer route is promoted: the only active Torch MLP-up backward rows flipped back to current C++ cuBLAS in focused uncontended reruns, and the tiny Triton GELU-forward row also flipped back to CUDA. | Target-context `RUN_LABEL=codex_sm120_round_torch_stack_refresh_20260521 RUN_TRAINING=0 BUILD_JOBS=4 RUN_PYTHON_STACK_BENCHMARKS=1 scripts/run_sm120_optimization_round.sh` validated with `benchmarks=187`, `stacks=9`, `family_stack_rows=168`, and `train_steps=0`. Initial selected rows included Torch fc dInput/dWeight `1346.50/1333.13 us` versus cuBLAS `1363.76/1353.38 us`, and Triton GELU forward `531.483 us` versus CUDA `535.954 us`; focused sequential reruns measured cuBLAS `1331.36/1332.63 us` versus Torch `1357.45/1347.02 us`, and CUDA GELU `528.159 us` versus Triton `529.627 us`. `promotion-candidates.json` now reports `0` active and `17` total candidates. |
| 2026-05-21 | cuBLAS dInput selector refinement | Rejected expanding the SM120 direct-cuBLAS dInput selector beyond the existing huge-N LM-head route. A narrow attention-projection/MLP-up selector passed correctness and won the 3-step smoke, but the x10 stability round regressed versus the current stable x10 default, so `llmc/matmul.cuh` was restored to the prior selector. | Candidate `./test_matmul` with added attproj/MLP-up dInput smoke rows passed `13/13` on the RTX 5090, and full 3-step round `codex_sm120_round_cublas_dinp_attproj_fc_20260521` validated with `avg_ms=2493.931`. The x10 stability round `codex_sm120_round_cublas_dinp_attproj_fc_x10_20260521` then validated but regressed to `avg_ms=2502.950` versus current stable x10 `2495.443 ms`. Refreshed benchmark evidence was mixed: qkv dInput did not justify broadening, attproj toggled around noise, and fc dInput favored direct cuBLAS only in the microbench. |
| 2026-05-21 | Current mixed-backend full smoke | Refreshed the current mixed-backend C++/CUDA/TK/cuBLAS/cuBLASLt path after the Torch, Triton, cuDNN, and CuTeDSL comparison work. The 3-step smoke keeps the trainer on the native mixed stack: Python operator rows remain benchmark/reference evidence unless they beat the native path and have a trainer-callable integration plan. | Target-context `RUN_LABEL=codex_sm120_round_current_mix_final_smoke_20260521 MAX_STEPS=3 BUILD_JOBS=4 RUN_PYTHON_STACK_BENCHMARKS=0 scripts/run_sm120_optimization_round.sh` passed correctness, benchmark, stack-probe, manifest, and training validation on the RTX 5090. The validator reports `benchmarks=93`, `stacks=9`, `family_stack_rows=168`, `train_steps=3`, and `avg_ms=2494.781`; final training step was `2496.65 ms` / `210150 tok/s` with final val loss `10.609911`. Checkpoint cleanup left only `DONE_00000003` and `main.log` in the train output directory. |
| 2026-05-21 | CuTeDSL GEMM artifact refresh | Restored CuTeDSL compile/rejection evidence in the active Python-stack selection round and taught `dev/validate_sm120_round.py` to emit `Unavailable Backend Rows`. CuTeDSL remains a candidate stack but has no timing row: the installed CUTLASS DSL grouped-GEMM path reaches a target-context compile attempt and rejects the local `sm_120a` BF16 route for every GPT-2 GEMM shape. | Target-context `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_cutedsl_matmul.py` captured in `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/bench_sm120_cutedsl_matmul.log` reports `CuTeDSL CUDA available: True`, device `NVIDIA GeForce RTX 5090`, and exact-shape unavailable rows for qkv, attproj, fc, fcproj, and LM-head with reason `local CuTeDSL BF16 grouped-GEMM path rejects sm_120a`. Revalidating the round passed with `benchmarks=180`, `stacks=9`, and `family_stack_rows=168`. |
| 2026-05-21 | Triton GEMM feasibility | Added `dev/triton/bench_sm120_matmul.py` and wired it into the optional Python-stack benchmark phase. The probe uses a generic BF16 Triton matmul kernel with row/column strides, 64-bit tensor offsets for LM-head-sized outputs, bias, approximate GeLU, dGeLU, and in-place accumulated dWeight variants. It is rejected for trainer promotion: every GPT-2 row is slower than the current cuBLAS/cuBLASLt/TK/Torch selection. | Target-context `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_matmul.py --repeats 2 --large-repeats 1 --output-rtol 0.05` captured in `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/bench_sm120_triton_matmul.log` measured qkv fwd/dInp/dW/dW+accum `2262.530/2504.180/2303.580/2316.410 us`, fc fwd+GeLU `3169.320 us`, fcproj dInp+dGeLU `3618.690 us`, and LM-head fwd/dInp/dW/dW+accum `49372.510/52579.710/75083.970/76736.100 us`. Revalidating the round passed with `benchmarks=180`, `stacks=9`, and `family_stack_rows=168`; `promotion-candidates.json` remains `0` active and `21` total. |
| 2026-05-21 | Triton attention forward feasibility | Added `dev/triton/bench_sm120_attention.py` and wired it into the optional Python-stack benchmark phase. The probe implements causal BF16 Triton forward kernels for both the GPT-2 separated-Q/K/V shape and the trainer-shaped packed-QKV layout, and records them as operator evidence only; backward is not implemented, and both forward rows are much slower than existing Torch, cuDNN, and packed TK rows. | Small target-context smoke `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_attention.py --batch 1 --seq 64 --channels 128 --heads 4 --repeats 1 --warmup 0 --block-m 16 --block-n 32` passed with separated/packed diffs `0.001953/0.003906`. GPT-2-sized `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_attention.py --repeats 3 --warmup 1` captured in `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/bench_sm120_triton_attention.log` measured Triton separated/packed forward `2113.485/2234.986 us`. Revalidating the round passed with `benchmarks=159`, `stacks=9`, and `family_stack_rows=168`. |
| 2026-05-20 | LayerNorm backward Python stacks | Extended `dev/triton/bench_sm120_layernorm.py` with LayerNorm backward comparisons. Torch native backward now benchmarks the full saved-mean/rstd backward (`dInput`, `dweight`, and `dbias`) against CUDA, while Triton reports both a dInput-only row and a full atomic FP32-gradient prototype. No trainer route is promoted: CUDA remains faster for full `C=768` backward, the dInput-only rows are not trainer-equivalent, and the Triton full row has an FP32 gradient-buffer contract mismatch. | Target-context GPT-2-shaped run in `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/bench_sm120_layernorm_python_stacks.log` measured `C=768` full Torch/Triton-atomic backward `416.224/363.680 us` versus CUDA `~290 us`, dInput-only Torch/Triton `217.568/228.320 us`, and `C=3072` full Torch/Triton-atomic backward `1395.008/1423.104 us`. Round revalidation passed with `benchmarks=156`, `stacks=9`, and `family_stack_rows=168`; `promotion-candidates.json` reports `0` active and `21` total candidates after the dInput-only row was resolved in the decision registry. |
| 2026-05-20 | Triton classifier feasibility | Added `dev/triton/bench_sm120_classifier.py` and wired it into the optional Python-stack benchmark phase. The probe implements tiled Triton log-sum-exp loss and BF16 dlogits rows for the GPT-2 padded-logits shape, using 64-bit tensor offset math so the 3.29B-element logits tensor is addressed correctly. It is rejected for trainer promotion: CUDA remains much faster on both loss-only and dlogits rows. | Small target-context smoke `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_classifier.py --batch 2 --seq 16 --vocab 257 --padded-vocab 320 --repeats 1 --warmup 0 --block-n 128` passed with both rows. GPT-2-sized `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_classifier.py --repeats 3 --warmup 1` captured in `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/bench_sm120_triton_classifier.log` measured Triton loss/dlogits `8379.872/21981.632 us`; same round CUDA measured `4354.061/9540.025 us`. Revalidating the round passed with `benchmarks=157`. |
| 2026-05-20 | cuDNN attention feasibility | Added `dev/bench_sm120_cudnn_attention.py` and wired it into the optional Python-stack benchmark phase. The probe uses the installed cuDNN 9.22 SDPA path through PyTorch's internal cuDNN attention op so cuDNN attention can be timed without adding `-lcudnn` to `train_gpt2cu`. No trainer route is promoted: native Torch SDPA is still the fastest already-separated Q/K/V reference row, cuDNNPacked forward is only a small focused edge over the current packed TK row, and cuDNNPacked backward is slower than TK. | Target-context `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_cudnn_attention.py --repeats 3 --warmup 1` captured in `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/bench_sm120_cudnn_attention.log` measured cuDNN separated forward/backward `667.986/2385.046 us` and cuDNNPacked forward/backward `780.912/3468.032 us`. Revalidating the round passed with `benchmarks=147`. |
| 2026-05-20 | cuDNN wheel-path stack probe | Fixed the optional-stack probe to find cuDNN headers and libraries from the active Python environment's `nvidia.cudnn` wheel layout, not only system CUDA paths. The Python-stack round now records cuDNN as `available` and marks it as a candidate for attention forward/backward, while preserving the no-`-lcudnn` v1 trainer contract. | `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/backend-stacks.md` found cuDNN `9.22.0` at `/home/adam/miniconda3/envs/llm-kittens/lib/python3.13/site-packages/nvidia/cudnn`. Revalidating the round passed with `stacks=9` and `family_stack_rows=168`. |
| 2026-05-20 | Decision-aware artifact refresh | Regenerated the Python-stack selection round artifacts with the decision registry enabled. The round persists `promotion-candidates.json`, `selected-backends.json`, and `scoreboard-candidates.md`; resolved non-trainer-callable winners stay in the full candidate list, while only rows without a registry decision remain active. The scoreboard writer now emits an explicit "no active promotion candidates" note when the active list is empty instead of an empty backlog table. | `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun --write-scoreboard scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/promotion-candidates.json --require-manifest --require-stack-probe --require-benchmarks` passed; after the LayerNorm backward refresh, `promotion-candidates.json` reports `0` active and `21` total promotion candidates. |
| 2026-05-20 | Active Torch backlog refresh | Refreshed the remaining active Torch rows from the Python-stack round. Rejected Torch qkv accumulated dWeight because current cuBLAS is faster. Kept Torch GELU forward and wide bias-add as operator evidence only: both still show tiny standalone Torch wins, but the edge is not large enough to justify a libtorch trainer route, and a native CUDA GELU block-size retune did not beat Torch. | Target-context `LLMK_BENCH_REPEATS=9 ./bench_sm120_runtime` measured CUDA `gelu_forward 546.950 us` and `bias_add OC=3072 549.777 us`; same-session Torch measured `538.895 us` and `547.969 us`. CUDA GELU block-size retunes measured `553.896 us` for `256` and `544.943 us` for `1024`. Target-context `LLMK_BENCH_REPEATS=9 ./bench_sm120_matmul` measured qkv `dW+accum cuBLAS 995.46 us`, while Torch qkv `dW+accum` measured `1095.54 us`. |
| 2026-05-20 | Promotion decision registry | Added `dev/sm120_promotion_decisions.json` and taught the validator to split active promotion candidates from resolved decisions. Torch is still selected where it wins a benchmark row, but stale or structurally non-promotable rows now carry `decision_status`/`decision_active` fields and move out of the active backlog once a same-session refresh rejects them. | `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --self-test` passed. Revalidating `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun` writes `promotion_candidates` with all selected non-trainer-callable rows and `active_promotion_candidates` only for rows that do not have a registry decision. |
| 2026-05-20 | Torch LM-head library refresh | Rechecked the highest-ranked Torch library-integration LM-head rows against a fresh C++ cuBLAS/cuBLASLt baseline. Rejected Torch LM-head integration for now: same-session C++ rows are equal or faster for forward, dInput, dWeight, and accumulated dWeight, so the old Torch LM-head promotion edge was stale benchmark-round evidence rather than a current reason to add libtorch or a native replacement. | Target-context `LLMK_BENCH_REPEATS=7 ./bench_sm120_matmul` measured LM-head `fwd cuBLASLt 22406.90 us`, `dInp cuBLAS 21338.28 us`, `dW cuBLASLt 20886.66 us`, and `dW+accum cuBLASLt 20932.98 us`. Same-session `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_matmul.py --shape lmhead --repeats 7 --large-repeats 5` measured Torch `22628.48/21383.14/21415.78/21621.12 us`. |
| 2026-05-20 | Promotion backlog class split | Split promotion backlog classes by implementation cost. Triton/native rows are now `native/codegen integration`, Torch operator wins that require libtorch or an equivalent native replacement are `library integration`, separated-Q/K/V Torch attention rows remain `layout rewrite`, and non-trainer LayerNorm widths stay `non-trainer shape`. This keeps "use Torch where it wins" visible without making a libtorch dependency look as cheap as a native CUDA/Triton route. | `dev/validate_sm120_round.py --self-test` covers class presence and ordering. Regenerated `/tmp/sm120_promotion_class_split_recheck.json` orders Triton/codegen rows before Torch `library integration` rows, then layout rewrites/reference gaps/non-trainer shapes. |
| 2026-05-20 | LayerNorm promotion-shape refresh | Rechecked the promotion-backlog Triton fused-residual LayerNorm `C=3072` row and closed a C++ benchmark coverage gap by adding `C=3072` to `bench_sm120_layernorm`. The row is rejected for trainer promotion: refreshed CUDA is slightly faster, and GPT-2 LayerNorm/fused residual uses hidden width `C=768`, not MLP width `C=3072`. Promotion candidates now classify LayerNorm rows with `C!=768` as `non-trainer shape` evidence. | Target-context `LLMK_BENCH_REPEATS=7 ./bench_sm120_layernorm` measured CUDA fused residual `C=768 279.785 us` and `C=3072 1101.192 us`; same-session `dev/triton/bench_sm120_layernorm.py --rows 65536 --cols 768 3072 --repeats 7 --warmup 3` measured Triton fused residual `C=768 313.856 us` and `C=3072 1121.344 us`. |
| 2026-05-20 | Torch attention feasibility refresh | Rechecked the top promotion-backlog Torch attention rows against the current packed TK trainer route. Native Torch SDPA over already-separated Q/K/V remains faster and useful as a Python/reference row, but the trainer-shaped `TorchPacked` path is slower than packed TK. No trainer integration is promoted; the promotion backlog now classifies separated-Q/K/V wins as `layout rewrite` rows and orders direct integration candidates first. | Same-session RTX 5090 `./bench_sm120_attention` measured packed TK forward/backward `787.859/2716.901 us`. `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_attention.py --repeats 7 --warmup 3` measured native Torch SDPA `570.848/2227.869 us`, but `TorchPacked` `1120.509/4107.011 us`. |
| 2026-05-20 | Triton GELU backward refresh | Rechecked the promotion-backlog Triton `gelu_backward_inplace` row against a fresh same-session CUDA baseline before starting any trainer integration. Rejected Triton promotion for this row: the current CUDA GELU backward path is faster in the refreshed target-context comparison, so the old promotion-backlog edge was cross-run noise/staleness rather than an integration signal. Promotion candidates now explicitly require a same-session baseline refresh before implementation work. | Target-context `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` on RTX 5090 measured CUDA `gelu_backward_inplace 789.605 us`; same-session `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_runtime.py --repeats 7 --warmup 3` measured Triton `802.830 us`. |
| 2026-05-20 | CUDA memory kernel comparison | Added `llmc/memory.cuh` and `CUDA kernel` rows to `bench_sm120_runtime` for the exact hidden/logits `cuda_memset` and `cuda_copy_d2d` shapes that Torch had selected as Python/operator wins. The native C++/CUDA route is rejected for trainer promotion: it is consistently slower than the existing CUDA runtime calls, so the Torch memory rows stay operator-prototype evidence rather than a reason to replace trainer memory calls. | Target-context `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` on RTX 5090 measured logits memset `CUDA runtime 4187.040 us` vs `CUDA kernel 4398.918 us`, logits copy `8988.563 us` vs `9391.521 us`, hidden memset `62.623 us` vs `64.806 us`, and hidden copy `133.621 us` vs `139.635 us`. |
| 2026-05-20 | Python-stack selected rows | Captured a real benchmark-only SM120 round with Torch/Triton/CuTeDSL stack probing and optional Python-stack benchmarks enabled. Torch is selected where it wins only for the stated scope: native SDPA is the reference route for already-separated Q/K/V, several LM-head and memory rows are operator prototypes, and Torch dlogits is unavailable at the full padded-logits shape due to OOM. CUDA/cuBLAS/cuBLASLt remain the trainer-compatible defaults unless a trainer-callable integration and TinyStories smoke prove otherwise. The validator now writes the same selections to `selected-backends.json` so future integration work can consume the per-row decision table directly. | Target-context `RUN_LABEL=codex_sm120_round_python_stacks_selection_20260520_rerun RUN_CORRECTNESS=0 RUN_TRAINING=0 RUN_BENCHMARKS=1 RUN_PYTHON_STACK_BENCHMARKS=1 RUN_STACK_PROBE=1 scripts/run_sm120_optimization_round.sh` passed with `benchmarks=143`, `stacks=9`, and `family_stack_rows=168`. Scoreboard: `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/scoreboard-candidates.md`; selected rows JSON: `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/selected-backends.json`. |
| 2026-05-20 | CuTeDSL GEMM feasibility | Added `dev/bench_sm120_cutedsl_matmul.py` to distinguish CuTeDSL package availability from an actual SM120 BF16 GEMM timing route. The local Torch/CUTLASS vendored grouped-GEMM CuTeDSL path currently rejects `sm_120a` for the BF16 `tcgen05` MMA path, so CuTeDSL remains a candidate stack with no timing row until a compatible SM120 dense/blockscaled kernel is scoped. | An earlier sandboxed smoke was not used as hardware-availability evidence; the target-context smoke reached CuTeDSL compilation and failed with `MmaF16BF16Op` expecting `sm_100*`/`sm_101*`/`sm_103*`, not `sm_120a`. `dev/bench_sm120_cutedsl_matmul.py` records every exact GPT-2 GEMM shape with that CuTeDSL rejection reason; the 2026-05-21 artifact refresh above is the current active-round evidence. |
| 2026-05-20 | LayerNorm backward blocks-per-SM | Rejected changing SM120 LayerNorm backward from `2 * SM` blocks to `1 * SM`. The candidate reduced focused backward timing by reducing cross-block partial-reduction work, but the x10 TinyStories stability round regressed versus the current default, so `llmc/layernorm.cuh` was restored to `blocks_per_sm=2`. `blocks_per_sm=3` was also rejected before a trainer round because the current scratch sizing is only safe for the existing two-blocks-per-SM allocation. | Default `./test_layernorm` passed on RTX 5090 and `LLMK_BENCH_REPEATS=7 ./bench_sm120_layernorm` measured `138.186/283.034/289.756 us` for forward/fused/backward. The `1 * SM` candidate passed `./test_layernorm` and measured `138.331/282.302/272.547 us`; after making it the temporary default, focused timing was `140.845/279.494/276.480 us`. Full round `codex_sm120_round_layernorm_bwd_blocks1_20260520` validated at `avg_ms=2492.310`, but x10 `codex_sm120_round_layernorm_bwd_blocks1_x10_20260520` validated at `avg_ms=2501.148`, slower than current-default x10 `2495.443 ms`. `3 * SM` passed `./test_layernorm` but `bench_sm120_layernorm` hit an illegal memory access. |
| 2026-05-20 | Stack probe interpreter alignment | Updated the SM120 round harness to use `PYTHON_BIN` when provided, otherwise the active `CONDA_PREFIX/bin/python` when available. This prevents optional Python backend probes from silently using the Homebrew `python3` path while the repo's `llm-kittens` conda env is active. The stack probe now records the Python executable and version in Triton/CuTeDSL evidence. | `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out /tmp/sm120_backend_stacks_conda.json --markdown-out /tmp/sm120_backend_stacks_conda.md` reports `Triton available` with `triton 3.6.0` and `CuTeDSL available` with `nvidia-cutlass-dsl 4.5.1` from Python `3.13.13`. `python -m py_compile dev/probe_sm120_backend_stacks.py dev/write_sm120_round_manifest.py` and `bash -n scripts/run_sm120_optimization_round.sh` passed. |
| 2026-05-20 | Triton LayerNorm forward prototype | Added a Python/Triton SM120 LayerNorm forward parity and timing prototype. It establishes that the conda-backed Triton stack can compile and run on the RTX 5090, but the naive row-wise forward kernel is not a CUDA baseline replacement at GPT-2 shapes, so it is not wired into the trainer. | `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_layernorm.py --rows 1024 --cols 768 3072 --repeats 3 --warmup 1` passed parity/timing on RTX 5090. GPT-2-sized timing with `--rows 65536 --cols 768 3072 --repeats 7 --warmup 3` measured Triton forward `178.240 us` for `C=768` and `575.840 us` for `C=3072`, with output max diff `0.031250` and exact mean/rstd versus the Torch FP32 reference. Current CUDA scoreboard rows remain faster at about `139.544 us` for `C=768` and `0.542 ms` for `C=3072`. |
| 2026-05-20 | Torch objective stack and first runtime rows | Added Torch as a first-class objective stack in the shared SM120 contract, optional-stack probe, validator parser, and round harness. The family/stack matrix now covers 168 rows, with Torch marked candidate for GEMM, attention, LayerNorm, and selected runtime families. Focused Torch LayerNorm/runtime prototypes were correct but did not produce a trainer-promotable win: native Torch LayerNorm is close but lacks saved mean/rstd, and stats-producing Torch compositions are far slower. | `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out /tmp/sm120_backend_stacks_torch.json --markdown-out /tmp/sm120_backend_stacks_torch.md` reports Torch `2.11.0+cu130` available from Python `3.13.13`. `dev/triton/bench_sm120_layernorm.py --rows 65536 --cols 768 3072 --repeats 7 --warmup 3` measured Torch native forward `153.088 us`/`556.384 us`, Torch stats forward `2222.848 us`/`9100.288 us`, and Torch fused stats `3226.208 us`/`13382.976 us`. Initial `dev/bench_sm120_torch_runtime.py --repeats 7 --warmup 3` rows covered bias add, GELU, and bias-gradient reductions; the later full runtime pass below supersedes those Torch runtime timings. Current CUDA rows remain faster or trainer-compatible. `bash -n scripts/run_sm120_optimization_round.sh`, py_compile, validator self-test, and a dry-run harness invocation passed. |
| 2026-05-20 | Torch GEMM matrix prototype | Added `dev/bench_sm120_torch_matmul.py`, covering the exact GPT-2 matmul shape matrix with Torch BF16 operators. The validator now ingests optional `bench_sm120_torch_matmul.log`, `bench_sm120_layernorm_python_stacks.log`, and `bench_sm120_torch_runtime.log` rows when present, and the SM120 harness writes the Torch matmul log in its Python-stack phase. No Torch GEMM row is promoted yet: most rows are slower, and the few marginal wins are too small to justify a libtorch trainer path without a stronger end-to-end hypothesis. | Torch all-shape run on RTX 5090 with `--repeats 7 --large-repeats 3` measured qkv `fwd/dInp/dW/dW+accum = 1448.81/1020.11/1005.34/1007.80 us`, attproj `522.64/371.21/337.84/343.97 us`, fc `fwd+GeLU/dInp/dW/dW+accum = 2441.34/1343.97/1364.82/1352.30 us`, fcproj `fwd/dInp/dInp+dGeLU/dW/dW+accum = 1583.71/1397.29/28754.41/1375.50/1398.93 us`, and lmhead `24324.06/21746.43/21517.34/21522.08 us`. Same-session native `LLMK_BENCH_REPEATS=7 ./bench_sm120_matmul` kept current cuBLAS/cuBLASLt/TK routes faster or close enough that no Torch promotion is justified. |
| 2026-05-20 | Torch attention SDPA prototype | Added `dev/bench_sm120_torch_attention.py`, covering native Torch SDPA over separated Q/K/V tensors and a trainer-shaped packed-QKV Torch path. Native SDPA is faster than the current TK packed-QKV microbenchmark and should be used as the reference row for already-separated Q/K/V Python-side experiments. It is not promoted into `train_gpt2cu`: the trainer-shaped packed Torch route is slower than TK once QKV view/layout and packed-gradient handling are included. | `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_attention.py --repeats 7 --warmup 3` on RTX 5090 measured native Torch SDPA forward/backward `571.850 us` / `2203.347 us`, but packed-QKV Torch forward/backward `1131.930 us` / `4092.714 us`. Same-session `./bench_sm120_attention` measured TK packed-QKV forward/backward `788.299 us` / `2723.085 us`. The validator now ingests optional `bench_sm120_torch_attention.log`, and the round harness writes it in the Python-stack phase. |
| 2026-05-20 | Torch classifier prototype | Added `dev/bench_sm120_torch_classifier.py`, covering the GPT-2 padded-logits classifier family with Torch BF16 `cross_entropy` loss and autograd dlogits rows. The first full-size attempt that forced FP32 logits materialization hit CUDA OOM, so the retained benchmark measures the feasible BF16 Torch route directly. It is rejected for trainer integration: Torch is much slower than the CUDA fused classifier on both loss-only and dlogits rows. | Small smoke `--batch 2 --seq 128 --repeats 3 --warmup 1` passed. GPT-2-sized `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_classifier.py --repeats 7 --warmup 3` measured Torch `fused_classifier_loss 18131.519 us` and `fused_classifier 34081.570 us`. Same-machine `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` measured CUDA `4031.207 us` and `9096.384 us`, so CUDA remains the trainer route. |
| 2026-05-20 | Torch runtime full-family prototype | Extended `dev/bench_sm120_torch_runtime.py` beyond bias/GELU to include Torch rows for global norm, AdamW, encoder, hidden/logits zeroing, and hidden/logits device copies. No trainer route is promoted. Torch fused AdamW with BF16 moment state is faster than CUDA, but it is not trainer-equivalent because the CUDA trainer uses FP32 moment buffers; the Torch FP32-state route is much slower. Torch memory operations are tie-range or slower, and Torch encoder/global-norm rows are slower. | `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_runtime.py --repeats 7 --warmup 3` measured Torch global norm `2366.618 us`, fused AdamW BF16-state `1225.190 us`, AdamW FP32-state `7452.800 us`, encoder `203.051 us`, hidden zero/copy `64.218/134.214 us`, and logits zero/copy `4397.152/9310.240 us`. Same-machine `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` measured CUDA global norm `185.434 us`, AdamW `1857.891 us`, encoder `87.607 us`, hidden zero/copy `64.508/133.980 us`, and logits zero/copy `4172.550/8968.173 us`. |
| 2026-05-20 | Triton runtime pointwise prototype | Added `dev/triton/bench_sm120_runtime.py` for Triton bias-add and GELU forward/backward rows at GPT-2 shapes. It passed parity against Torch references and is wired into the optional Python stack benchmark phase. No Triton runtime row is promoted: bias add and GELU are slower than same-machine CUDA rows, though GELU backward is much closer than the composed Torch backward row. | Small smoke `--rows 1024 --repeats 3 --warmup 1` passed on RTX 5090. Full `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_runtime.py --repeats 7 --warmup 3` measured Triton bias add `146.328/573.555 us` for `OC=768/3072`, GELU forward `573.830 us`, and GELU backward `840.723 us`. Same-machine CUDA rows were `82.589/537.874 us` for bias add and `537.595/791.320 us` for GELU forward/backward. |
| 2026-05-20 | Bias-gradient grid-y cap     | Rejected capping SM120 `matmul_backward_bias` `grid_size_y` to reduce auxiliary reduction work. Caps `8` and `16` preserved correctness but did not improve the three trainer-active GPT-2 bias-gradient widths as a set, so the temporary launch-policy hook was removed and the existing 512-thread occupancy-derived policy remains unchanged. | Default `make -B -j 4 test_bias bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` built and `./test_bias` passed on RTX 5090. Default `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` measured `bias_grad_reduce 24.344/188.189/247.656 us` for `OC=768/2304/3072`. Cap `8` passed `./test_bias` but measured `32.888/190.440/246.306 us`; cap `16` passed `./test_bias` but measured `24.342/189.530/247.672 us`. |
| 2026-05-20 | Runtime shape validator      | Added runtime shape coverage to `dev/validate_sm120_round.py`, backed by `RUNTIME_SHAPE_REQUIREMENTS` in `dev/sm120_objective_contract.py`. Full SM120 rounds now fail if required runtime rows such as bias add `OC=768/3072` or bias-gradient reduction `OC=768/2304/3072` disappear from `bench_sm120_runtime.log`. | `python3 dev/validate_sm120_round.py --self-test` passed, including a new negative synthetic round that omits `bias_grad_reduce BT=65536 OC=2304`. Full round `codex_sm120_round_runtime_shape_validator_20260520` passed with `benchmarks=84`, `family_stack_rows=147`, `train_steps=3`, `avg_ms=2545.384`, runtime shape coverage all true, and checkpoint cleanup verified. |
| 2026-05-20 | Runtime memory shape coverage | Expanded `bench_sm120_runtime` and `RUNTIME_SHAPE_REQUIREMENTS` so future full rounds must report both hidden-state and padded-logits memory overhead rows for `cuda_memset` and `cuda_copy_d2d`. This closes a benchmark blind spot for profiler-visible runtime overhead around the classifier-sized activation. | `python3 -m py_compile dev/sm120_objective_contract.py dev/validate_sm120_round.py` and `python3 dev/validate_sm120_round.py --self-test` passed. `make -B -j 4 bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` passed. Focused `LLMK_BENCH_REPEATS=5 ./bench_sm120_runtime` on RTX 5090 measured logits `cuda_memset 4181.555 us`, logits `cuda_copy_d2d 8915.744 us`, hidden `cuda_memset 62.980 us`, hidden `cuda_copy_d2d 133.756 us`. Full round `codex_sm120_round_memory_shape_coverage_20260520` then passed on RTX 5090 with `benchmarks=86`, `family_stack_rows=147`, `train_steps=3`, `avg_ms=2490.206`, GPU runtime `available`, all runtime memory shapes covered, and checkpoint cleanup verified. |
| 2026-05-20 | Current default x10 refresh | Refreshed the current mixed-backend default after the runtime memory-shape coverage work. The x10 round is the new stable current-default evidence: it keeps the same shape-aware selector policy, covers the expanded 86-row benchmark contract, and narrowly beats the previous same-harness x10 run. | Full round `codex_sm120_round_current_default_x10_after_memory_20260520` passed on RTX 5090 with `benchmarks=86`, `family_stack_rows=147`, `train_steps=10`, `avg_ms=2495.443`, GPU runtime `available`, all objective/GEMM/provider/runtime-shape coverage true, and checkpoint cleanup verified. Previous current-default x10 was `codex_sm120_round_cublas_bwd_default_x10_20260520` at `2497.246 ms`. |
| 2026-05-20 | AdamW no-master specialization | Rejected compile-time specialization of the no-master AdamW update path. The candidate removed the per-element `master_params_memory != NULL` branch and improved one same-session focused AdamW timing, but the 10-step trainer stability round regressed versus the current default x10 evidence, so `llmc/adamw.cuh` was restored to the existing single 512-thread CUDA kernel. | Candidate `./test_adamw` passed on RTX 5090. Same-session `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` measured specialized `adamw_update 1831.962 us` versus disabled old-path `1851.027 us`. Full 3-step round `codex_sm120_round_adamw_nomaster_specialized_20260520` validated at `avg_ms=2494.281`, but x10 `codex_sm120_round_adamw_nomaster_specialized_x10_20260520` validated at `avg_ms=2499.597`, slower than current-default x10 `2495.443 ms`. Final restored rebuild passed `./test_adamw` and measured `adamw_update 1841.731 us`. |
| 2026-05-20 | Classifier loss log-sum-exp formula | Rejected replacing the target `expf()` plus `logf()` loss calculation with the algebraically equivalent `offset - target_logit - log(scale)` formula. The candidate passed correctness but did not prove a focused runtime win: it improved the dlogits row only within noise while regressing the loss-only row, so the source was restored before any trainer round. | Candidate `./test_fused_classifier` passed on RTX 5090. Candidate `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` measured `fused_classifier_loss 4027.225 us` and `fused_classifier 9084.755 us`; restored same-session rebuild passed `./test_fused_classifier` and measured `3979.443 us` / `9103.584 us`. Current stable x10 runtime row remains `fused_classifier_loss 3972.230 us` and `fused_classifier 8985.197 us`. |
| 2026-05-20 | Classifier loss-only block-size sweep | Rejected separate SM120 block-size tuning for `fused_classifier<WriteDLogits=false>`. A default-preserving `LLMK_SM120_CLASSIFIER_LOSS_ONLY_BLOCK_SIZE` hook now allows future loss-only A/B without perturbing the training dlogits block size, but tested values did not beat the stable loss-only row, so the default remains tied to `LLMK_SM120_CLASSIFIER_BLOCK_SIZE=1024`. | `-DLLMK_SM120_CLASSIFIER_LOSS_ONLY_BLOCK_SIZE=512`, `256`, and `768` each built and passed `./test_fused_classifier` on RTX 5090. Focused `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` measured loss-only rows `3972.480 us`, `5193.088 us`, and `4022.528 us`; the stable x10 default row remains `3972.230 us`. Final default rebuild passed `./test_fused_classifier`. |
| 2026-05-20 | Classifier loss-sync elision | Rejected moving the training-path loss write into the dlogits loop to remove the pre-overwrite `__syncthreads()`. The candidate preserved correctness but did not beat the restored pre-sync loss path in same-session focused timing, so the source behavior remains unchanged. | Candidate `make -B -j 4 test_fused_classifier bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` built; candidate `./test_fused_classifier` passed on RTX 5090. Candidate `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` measured `fused_classifier 9097.907 us`; restored pre-sync A/B build measured `9063.795 us`. Final restored source rebuild passed `./test_fused_classifier`. |
| 2026-05-20 | Global norm block-size sweep | Rejected SM120 global-norm block-size-only tuning. Added a scoped `LLMK_SM120_GLOBAL_NORM_BLOCK_SIZE` compile hook with the existing 512-thread default, then tested 256 and 1024. The 256-thread row matched the stable baseline range rather than proving a win, and 1024 regressed badly, so the default remains 512. | Default `make -B -j 4 test_global_norm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` built and `./test_global_norm` passed on RTX 5090. Same-session `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` measured default `global_norm_squared 199.866 us` under a noisy run, `-DLLMK_SM120_GLOBAL_NORM_BLOCK_SIZE=256` at `186.139 us`, and `1024` at `315.776 us`; current stable x10 default is `185.869 us`. Final restored default rebuild passed `./test_global_norm`. |
| 2026-05-20 | LayerNorm smem attribute cache | Rejected caching `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` in the CUDA LayerNorm launchers. The candidate preserved focused correctness and a 3-step round, but the 10-step stability round regressed badly, so the source was restored to the existing per-call attribute setup. | Candidate `./test_layernorm` passed on RTX 5090. Same-session focused timing with the cache measured `141.641/279.668/289.124 us` for forward/fused/backward versus cache-disabled `140.892/279.188/290.593 us`. Full round `codex_sm120_round_layernorm_smem_attr_cache_20260520` passed at `avg_ms=2494.965`, but x10 `codex_sm120_round_layernorm_smem_attr_cache_x10_20260520` passed validation at `avg_ms=2630.194`, slower than current-default x10 `2495.443 ms`. |
| 2026-05-20 | Classifier padded-tail zeroing | Rejected explicit zeroing for classifier dlogits padding columns `[V, P)`. The candidate strengthened the standalone kernel contract and passed focused correctness, but the full TinyStories round regressed, so the classifier source and smoke test were restored to the existing trainer invariant where padded embedding rows keep padded logits zero. | Candidate `./test_fused_classifier` passed on RTX 5090 and same-session focused runtime measured `fused_classifier 9019.360 us`. A disabled-tail-zero A/B failed the strengthened smoke (`dlogits max abs diff = 1.992188`) and measured `9082.880 us`. Full round `codex_sm120_round_classifier_pad_tail_zero_20260520` passed validation with `benchmarks=86`, `family_stack_rows=147`, `train_steps=3`, `avg_ms=2596.574`, slower than current-default x10 `2495.443 ms`. |
| 2026-05-20 | Bias-gradient shape coverage | Expanded `bench_sm120_runtime` bias-gradient coverage from the single MLP `OC=3072` row to the trainer-active GPT-2 bias widths: attention/projection `OC=768`, qkv `OC=2304`, and MLP `OC=3072`. This is benchmark matrix coverage, not a source-kernel promotion. | `make -B -j 4 test_bias bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` passed. `./test_bias` passed on RTX 5090. `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` measured bias-gradient rows at `25.643 us`, `188.067 us`, and `262.192 us` for `OC=768/2304/3072`. |
| 2026-05-20 | Bias add vec2                | Promoted an SM120 standalone `add_bias` vectorized CUDA path for aligned row widths. Each thread handles two adjacent `x128` BF16 packs; unaligned widths keep the scalar fallback. This improves the runtime-family bias-add rows but is not a TinyStories speed promotion because the default SM120/cuBLASLt trainer forward path fuses bias in the matmul epilogue. | Default `./test_bias` passed on RTX 5090 for hidden aligned, MLP aligned, and unaligned fallback shapes. Default `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` measured `bias_add 93.706 us / 556.463 us`; same-session scalar rebuild with `EXTRA_NVCC_FLAGS=-DLLMK_SM120_DISABLE_BIAS_ADD_VEC2` measured `112.179 us / 596.032 us`. |
| 2026-05-20 | Encoder forward vec2         | Rejected an SM120 encoder-forward vec2 CUDA path where each thread handled two adjacent `x128` packs to halve token-id reloads and block count for `C=768`. The candidate passed correctness but regressed focused runtime timing, so the source was restored. | Candidate `./test_encoder` passed on RTX 5090 with max abs diff `0.000488`. Candidate `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` measured `encoder_forward 88.138 us`; restored source measured `87.254 us` under the same repeat count. |
| 2026-05-20 | GELU block-size sweep        | Rejected SM120 GELU block-size-only tuning. Forward block size `256` matched a same-session default rerun rather than proving a real win, and backward block size `256` regressed the backward row. Keep the current SM120 defaults: forward block `512`, backward block `128`. | Default `make -B -j 4 test_gelu bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` built and `./test_gelu` passed on RTX 5090. Default `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` measured `gelu_forward 551.834 us`, `gelu_backward_inplace 799.930 us`; `-DLLMK_SM120_GELU_BWD_BLOCK_SIZE=256` measured `535.692/807.368 us`; `-DLLMK_SM120_GELU_FWD_BLOCK_SIZE=256` measured `535.210/798.405 us`; default rerun measured `535.383/798.040 us`. |
| 2026-05-20 | GELU backward sech rewrite   | Rejected a GELU backward algebraic rewrite that replaced `coshf(arg)` plus reciprocal square with `1 - tanh(arg)^2`. The candidate passed the independent `test_gelu` CPU-reference smoke, but the focused runtime benchmark did not improve `gelu_backward_inplace`, so the source was restored. | Candidate `./test_gelu` passed on RTX 5090 with backward max abs diff `0.003887`. Candidate `LLMK_BENCH_REPEATS=5 ./bench_sm120_runtime` measured `gelu_backward_inplace 799.662 us`, matching the restored/default range (`795-800 us`) rather than improving it. |
| 2026-05-20 | Global norm reset elision    | Rejected a global-norm reset-memset elision candidate. The source candidate skipped the reset memset when the partial-sum buffer was fully covered by the norm kernel and fell back to the existing memset path for tail cases. Correctness passed, but focused runtime timing was flat versus the restored source, so the source hook was removed. Kept the reset-tail smoke coverage in `test_global_norm`. | Candidate `LLMK_BENCH_REPEATS=5 ./bench_sm120_runtime` measured `global_norm_squared 185.819 us`; restored source measured `185.605 us`. `./test_global_norm` passed on RTX 5090, including the stale-tail reset case (`reset-tail ... relative diff = 0.000000 PASS`). |
| 2026-05-20 | Classifier loss-only path    | Accepted an SM120 fused-classifier loss-only early return for validation/forward-loss calls. The `WriteDLogits=true` training path stays unchanged; this is a validation/runtime micro-op improvement, not a stable training-speed promotion. The 3-step round looked faster, but the 10-step stability run did not beat the current default x10 evidence. | `./test_fused_classifier` passed loss-only loss, untouched-logits, loss, and dlogits checks. Focused `LLMK_BENCH_REPEATS=5 ./bench_sm120_runtime` measured `fused_classifier_loss 4037.069 us` versus `fused_classifier 9054.438 us`. Full rounds `codex_sm120_round_classifier_loss_only_20260520` and `codex_sm120_round_classifier_loss_only_x10_20260520` passed validation at `avg_ms=2488.125` and `2551.284`; the x10 run is slower than `codex_sm120_round_cublas_bwd_default_x10_20260520` at `2497.246 ms`. |
| 2026-05-20 | Classifier loss-only online softmax | Rejected a scoped SM120 one-pass online softmax prep for `WriteDLogits=false` only. It kept the training dlogits path unchanged and passed correctness, but focused same-session runtime regressed both the loss-only and dlogits classifier rows, so the source was restored to the current two-pass SM120 prep. | Candidate `make -B -j 4 test_fused_classifier bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` built; candidate `./test_fused_classifier` passed on RTX 5090. Candidate `LLMK_BENCH_REPEATS=7 ./bench_sm120_runtime` measured `fused_classifier_loss 4356.179 us` and `fused_classifier 9792.198 us`; restored source measured `3943.974 us` and `9111.564 us` under the same repeat count. |
| 2026-05-20 | Runtime benchmark repeatability | Changed `bench_sm120_runtime` to report the median of repeated event samples, with `LLMK_BENCH_REPEATS=<n>` as an override. Runtime-family decisions now use repeated samples like matmul and LayerNorm instead of single-shot timings. | `make -B -j 4 test_fused_classifier bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` passed. Default focused `LLMK_BENCH_REPEATS=5 ./bench_sm120_runtime` before the loss-only specialization measured `fused_classifier 10284.915 us`; after the loss-only specialization, the x10 round measured `fused_classifier_loss 3978.829 us` and `fused_classifier 9090.682 us`. |
| 2026-05-20 | LayerNorm forward no-smem selector | Rejected the opt-in SM120 LayerNorm forward no-shared-memory route. Median focused timing initially showed a forward-only row win, but the full TinyStories round regressed badly, so the temporary selector hook was removed and the shared-memory CUDA forward path remains default. The broader fused-residual no-smem fallback was also rejected before a trainer round because it regressed fused residual forward timing. | Focused default `LLMK_BENCH_REPEATS=5 ./bench_sm120_layernorm` measured forward/fused/backward `149.361/279.337/285.874 us`; opt-in `-DLLMK_SM120_LAYERNORM_NO_SMEM` measured `138.172/279.618/291.385 us`. Full `RUN_LABEL=codex_sm120_round_layernorm_forward_nosmem_20260520 MAX_STEPS=3 BUILD_JOBS=4 EXTRA_NVCC_FLAGS=-DLLMK_SM120_LAYERNORM_NO_SMEM scripts/run_sm120_optimization_round.sh` passed validation but regressed to `avg_ms=2662.431`, with round-local LayerNorm `151.580/307.380/315.872 us`. |
| 2026-05-20 | LayerNorm benchmark repeatability | Changed `bench_sm120_layernorm` to report the median of repeated event samples, with `LLMK_BENCH_REPEATS=<n>` as an override. Single-shot LayerNorm measurements swung enough to flip the apparent no-smem forward winner, so LayerNorm selector decisions now require repeated-sample focused timing plus trainer evidence. | `make -B -j 4 test_layernorm bench_sm120_layernorm train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` passed; restored default `./test_layernorm` passed on RTX 5090; `LLMK_BENCH_REPEATS=5 ./bench_sm120_layernorm` reported `Timing: median of 5 event samples per row` and default timings `141.911/282.536/289.764 us`. |
| 2026-05-20 | Classifier target-logit cache | Rejected an opt-in SM120 fused-classifier target-logit cache that removed the post-loss block sync. The candidate passed correctness, but the focused runtime benchmark regressed the classifier row, so the temporary hook was removed.                                                                                                                                                                                        | `make -B -j 4 test_fused_classifier bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1 EXTRA_NVCC_FLAGS=-DLLMK_SM120_CLASSIFIER_CACHE_TARGET_LOGIT` built; `./test_fused_classifier` passed; `./bench_sm120_runtime` measured fused_classifier `9963.085 us`, slower than the current ~`9.0 ms` default range. |
| 2026-05-20 | TK fused dGELU dInput selector | Rejected the opt-in SM120 TK fused dGELU dInput route before a trainer round. The focused benchmark had shown a small row win, but the trainer-shaped `C=3072, OC=768` smoke failed the existing matmul correctness threshold, so the temporary selector hook was removed and the current cuBLASLt DGELU epilogue remains the trainer route.                                                                                  | `make -B -j 4 test_matmul DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1 EXTRA_NVCC_FLAGS=-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP` built; `./test_matmul` on RTX 5090 failed only `dInp backward fused dGELU (GPT-2 fcproj route)` with max abs diff `0.500000` against the strict `< 0.50` gate. |
| 2026-05-20 | LM-head forward cuBLAS selector | Rejected the opt-in SM120 LM-head forward direct-cuBLAS route. The scoped macro build passed focused matmul correctness and the full round validation gates, but TinyStories regressed versus the current default, so the temporary forward selector hook was removed and LM-head forward stays on cuBLASLt in the trainer.                                                                                                      | `RUN_LABEL=codex_sm120_round_cublas_lmhead_forward_20260520 ... EXTRA_NVCC_FLAGS=-DLLMK_SM120_USE_CUBLAS_LMHEAD_FORWARD scripts/run_sm120_optimization_round.sh` passed validation with `benchmarks=81`, `family_stack_rows=147`, `train_steps=3`, `avg_ms=2522.413`. Round-local LM-head forward was cuBLAS `22186.19 us` vs cuBLASLt `23528.42 us`, but trainer timing regressed versus the current default `2495-2497 ms`. |
| 2026-05-20 | qkv forward TK selector      | Rejected the opt-in SM120 qkv-forward TK route. The scoped macro build passed all correctness and round validation gates, but the repeated-sample benchmark did not reproduce a qkv forward win and the 3-step TinyStories smoke regressed badly versus the current default, so the temporary selector hook was removed.                                                                                                         | `RUN_LABEL=codex_sm120_round_tk_qkv_forward_20260520 ... EXTRA_NVCC_FLAGS=-DLLMK_SM120_USE_TK_QKV_FORWARD scripts/run_sm120_optimization_round.sh` passed validation with `benchmarks=81`, `family_stack_rows=147`, `train_steps=3`, `avg_ms=2598.442`. Round-local qkv forward was TK `1073.24 us` vs cuBLASLt `1039.15 us`; current default remains `2495-2497 ms`. |
| 2026-05-20 | Matmul benchmark repeatability | Changed `bench_sm120_matmul` to report the median of repeated event samples instead of a single event measurement. Single-shot cuBLASLt timings were flipping apparent qkv/fcproj forward winners between adjacent runs, which is too weak for shape-selector promotion. Keep current selectors until a repeated-sample focused benchmark plus trainer round proves a change.                                                                 | `make -B -j 4 bench_sm120_matmul DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` passed. Rebuilt `./bench_sm120_matmul` on RTX 5090 reported `Timing: median of 3 event samples per provider`; median rows included qkv fwd TK `1072.64 us` vs cuBLASLt `1091.65 us`, fcproj fwd TK `1419.81 us` vs cuBLASLt `1368.16 us`, and fcproj `dInp+dGeLU` TK `1789.54 us` vs cuBLASLt fused `1802.00 us`. |
| 2026-05-20 | Classifier online softmax     | Rejected a one-pass online SM120 softmax-prep traversal for fused classifier. It removed one logits read during max/sum preparation and passed correctness, but regressed the runtime benchmark versus the restored two-pass 1024-thread default, so the source was restored to the two-pass traversal.                                                                                                                               | Online candidate: `make -B -j 4 test_fused_classifier bench_sm120_runtime DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`, `./test_fused_classifier` passed, `./bench_sm120_runtime` measured fused_classifier `9140.295 us`. Restored default rebuild passed the same smoke and measured `9086.694 us`.                                                                                       |
| 2026-05-20 | Current default x10 timing    | Replaced the historical `2469 ms` note with same-harness current-default evidence. The promoted direct-cuBLAS backward selector validated over 10 TinyStories steps at `2497.246 ms`; the run is stable and faster than the restored `2508.27 ms` baseline, but it does not reproduce the older `2469 ms` timing. Treat the historical note as stale/non-comparable until a future same-harness round beats it. | `RUN_LABEL=codex_sm120_round_cublas_bwd_default_x10_20260520 MAX_STEPS=10 BUILD_JOBS=4 scripts/run_sm120_optimization_round.sh` passed validation with `benchmarks=81`, `family_stack_rows=147`, `train_steps=10`, `avg_ms=2497.246`; per-step timings were `2495.40`, `2488.70`, `2491.16`, `2493.62`, `2495.75`, `2495.22`, `2503.11`, `2499.24`, `2501.50`, `2506.93 ms`. |
| 2026-05-20 | Classifier block-size sweep   | Rejected SM120 fused-classifier block sizes `256`, `512`, and `768`. All three passed the focused classifier correctness smoke, but each regressed the runtime benchmark versus the restored 1024-thread default, so keep `LLMK_SM120_CLASSIFIER_BLOCK_SIZE=1024`.                                                                                                                                                              | `LLMK_SM120_CLASSIFIER_BLOCK_SIZE=256` passed `./test_fused_classifier` but measured `12782.669 us`; `512` passed and measured `9147.154 us`; `768` passed and measured `9160.519 us`; default rebuild measured `9090.138 us` in `./bench_sm120_runtime`.                                                                                             |
| 2026-05-20 | cuBLAS backward GEMM selector | Promoted a shape-aware SM120 direct-cuBLAS backward GEMM selector inside the cuBLASLt fallback path. The default now uses direct cuBLAS for large-OC non-fused dInput and for qkv/attention-projection/MLP dWeight rows where cuBLAS repeatedly beat cuBLASLt in focused timing. Fused dGELU and LM-head dWeight stay on cuBLASLt. One default round was noisy at `2542.302 ms`, so the promotion relies on the opt-in run plus the default rerun. | Focused `./test_matmul` passed `10/10` including route-specific dInput and dWeight cases; focused `./bench_sm120_matmul` showed cuBLAS wins for dWeight rows and large-OC dInput; `RUN_LABEL=codex_sm120_round_cublas_bwd_20260520 ... EXTRA_NVCC_FLAGS=-DLLMK_SM120_USE_CUBLAS_BACKWARD_GEMM` validated at `avg_ms=2493.738`; default rerun `codex_sm120_round_cublas_bwd_default_rerun_20260520` validated at `avg_ms=2495.261`. |
| 2026-05-20 | AdamW no-master launch       | Rejected the SM120 AdamW 1024-thread/no-master launch experiment. Focused timings were noisy and the full TinyStories round regressed to `2567.526 ms` versus the retained 1024-thread classifier round at `2528.193 ms`, so the AdamW runtime default was reverted to the existing 512-thread CUDA path. Keep the added no-master correctness coverage in `test_adamw`.                                                                                                                               | `./test_adamw` passed master and no-master checks; focused `./bench_sm120_runtime` saw AdamW around `1815 us`, but full `RUN_LABEL=codex_sm120_round_adamw1024_20260520 MAX_STEPS=3 BUILD_JOBS=4 scripts/run_sm120_optimization_round.sh` passed validation with `avg_ms=2567.526` and AdamW `1887.088 us`, so it was rejected.                        |
| 2026-05-20 | Fused classifier SM120 softmax | Replaced the fragile SM120 classifier softmax traversal with a two-pass vectorized row max/sum path and classifier-local reductions, allowing a 1024-thread launch on RTX 5090. This supersedes the 64-thread correctness workaround: focused classifier timing improved from `13062.177 us` to `9086.464 us`, and the full TinyStories round improved from `2577.055 ms` to `2528.193 ms`. It is still not a speed promotion over the documented `2508.27 ms` baseline. | `timeout 20s ./test_fused_classifier` passed with loss diff `0.000001` and dlogits diff `0.000173`; focused `./bench_sm120_runtime` recorded fused classifier `9064.749 us`; full `RUN_LABEL=codex_sm120_round_classifier_1024_20260520 MAX_STEPS=3 BUILD_JOBS=4 scripts/run_sm120_optimization_round.sh` passed with validator `avg_ms=2528.193`. |
| 2026-05-20 | Full SM120 round             | Validated the current mixed backend after fixing LayerNorm and fused-classifier correctness gates. The round is not a speed promotion: TinyStories averaged `2577.055 ms`, slower than the documented `2508.27 ms` restored baseline and the historical `2469 ms` note, so keep the current strategy and use this row as correctness/coverage evidence.                                                                                                                                        | `RUN_LABEL=codex_sm120_round_after_lnfix_clsfix_20260520 MAX_STEPS=3 BUILD_JOBS=4 scripts/run_sm120_optimization_round.sh` passed; validator reported `benchmarks=81`, `family_stack_rows=147`, `train_steps=3`, `avg_ms=2577.055`; artifacts in `scratch/sm120_rounds/codex_sm120_round_after_lnfix_clsfix_20260520`.                                |
| 2026-05-20 | LayerNorm backward           | Fixed the SM120 LayerNorm backward accumulator for partial blocks. Inactive warps now enter each block reduction round and contribute explicit zeros instead of leaving stale shared-memory partials for `dbias`/`dweight`.                                                                                                                                                                                                                                                                     | Focused `./test_layernorm` on RTX 5090 changed `backward dbias` from `1.510559 FAIL` to `0.001953 PASS`; full round `codex_sm120_round_after_lnfix_clsfix_20260520/test_layernorm.log` passed.                                                                                                                                                        |
| 2026-05-20 | Fused classifier SM120 gate  | Re-enabled runtime coverage on SM120 without a skip by using a 64-thread classifier launch for `KITTENS_SM120`. The 128-thread launch hung on RTX 5090; 64 threads passes correctness but is visibly slower in the runtime benchmark, so the next optimization target is an SM120-safe multi-warp classifier path or a stack-specific replacement.                                                                                                                                              | `timeout 20s ./test_fused_classifier` passed at 64 threads with loss diff `0.000001` and dlogits diff `0.000173`; 128-thread test timed out; full round runtime benchmark recorded fused classifier `13062.177 us`.                                                                                                                                    |
| 2026-05-20 | Shared objective contract    | Centralized the SM120 allowed stacks, required timing families, runtime kernels, correctness targets, and manifest binary list in `dev/sm120_objective_contract.py`. The stack probe, manifest writer, and round validator now import the same contract, so a round cannot drift because one tool was updated and another still enforces an older matrix.                                                                                                                                           | `python3 -m py_compile dev/sm120_objective_contract.py dev/probe_sm120_backend_stacks.py dev/validate_sm120_round.py dev/write_sm120_round_manifest.py`; `python3 dev/validate_sm120_round.py --self-test`; `python3 dev/probe_sm120_backend_stacks.py --json-out /tmp/sm120_backend_stacks_shared_contract.json --markdown-out /tmp/sm120_backend_stacks_shared_contract.md`. |
| 2026-05-20 | Manifest binary identity     | Tightened manifest validation so all expected correctness, benchmark, and trainer executables must be present and hashed. This prevents a round from passing with only a partial binary manifest after a stale or missing build artifact.                                                                                                                                                                                                                                                        | `python3 dev/validate_sm120_round.py --self-test` covers the positive full manifest and a missing-`bench_sm120_runtime` negative case.                                                                                                                                                                                                                  |
| 2026-05-20 | Baseline provider coverage   | Added a validator gate that cross-checks parsed benchmark rows against the baseline provider recorded in the family-stack matrix. A round can no longer satisfy a family by name alone if, for example, `adamw` is parsed under a non-CUDA stack or attention forward is missing its TK baseline row. The stack matrix also now records SM120 TK LayerNorm as missing rather than candidate, because the current TK LayerNorm wrapper is Hopper-only and the SM120 wrapper routes through CUDA. | `python3 dev/validate_sm120_round.py --self-test` covers the positive baseline-provider matrix and a wrong-AdamW-provider negative case; `python3 dev/probe_sm120_backend_stacks.py --json-out /tmp/sm120_backend_stacks_baseline_provider.json --markdown-out /tmp/sm120_backend_stacks_baseline_provider.md` records the source-aware LayerNorm row. |
| 2026-05-20 | Family-stack applicability   | Extended `backend-stacks.json` with a family/stack matrix covering every required objective family against ThunderKittens, cuBLAS, cuBLASLt, cuDNN, Triton, Torch, CuTeDSL, and Plain CUDA. The current matrix is 168 rows. The validator now rejects missing matrix rows and writes a `Backend Family-Stack Matrix` summary, so unsupported pairings such as cuBLAS attention or TK runtime kernels are recorded as not applicable with reasons.                                                                              | `python3 dev/probe_sm120_backend_stacks.py --json-out /tmp/sm120_backend_stacks_matrix.json --markdown-out /tmp/sm120_backend_stacks_matrix.md` wrote the matrix; `python3 dev/validate_sm120_round.py --self-test` covers the positive matrix and a missing family-stack row negative case.                                                           |
| 2026-05-20 | cuBLAS provider coverage     | Extended `bench_sm120_matmul` with direct cuBLAS timing rows for plain forward, dInput, dWeight, and accumulated dWeight, plus explicit cuBLAS+CUDA rows for forward+GELU and dInput+dGELU. The validator now writes `GEMM Provider Coverage` at pass+shape granularity and rejects benchmark logs that only compare TK against cuBLASLt, even if just one required shape is missing cuBLAS.                                                                                                    | `python3 dev/validate_sm120_round.py --self-test` covers the positive synthetic matrix and a missing-LM-head-cuBLAS negative case; `make -j 4 bench_sm120_matmul DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` passed.                                                                                                                                |
| 2026-05-20 | Stack matrix coverage        | Expanded the stack probe to match the allowed-stack list in `optimise-goal.md` and made the validator reject incomplete probe JSON. Missing stacks can be marked `missing` or `blocked`, but every allowed stack now needs an evidence row and next action before a round can pass `--require-stack-probe`.                                                                                                                                                                                     | `python3 dev/validate_sm120_round.py --self-test` covers the positive synthetic stack matrix and a missing-CuTeDSL negative case.                                                                                                                                                                                                                      |
| 2026-05-20 | GEMM shape coverage          | Tightened the SM120 round validator so benchmark coverage is not satisfied by a single representative GEMM row. It now requires the explicit GPT-2 shape matrix from `optimise-goal.md` and writes a `GEMM Shape Coverage` table in `scoreboard-candidates.md`.                                                                                                                                                                                                                                 | `python3 dev/validate_sm120_round.py --self-test` covers the positive synthetic round and a missing-LM-head negative case.                                                                                                                                                                                                                             |
| 2026-05-20 | Objective coverage           | Extended the SM120 artifact validator to fail benchmark rounds that do not cover all required objective families: GEMM fwd/dInput/dWeight/accum/fused epilogues, attention fwd/bwd, LayerNorm variants, classifier, AdamW, global norm, encoder, memset, and copy.                                                                                                                                                                                                                              | `python3 dev/validate_sm120_round.py --self-test` passed with positive and missing-family synthetic cases.                                                                                                                                                                                                                                             |
| 2026-05-20 | Evidence manifest            | Added `dev/write_sm120_round_manifest.py` and made the round validator require `round-manifest.json` for non-dry harness runs, giving candidate rows stable config, commit, toolchain, and binary SHA256 evidence.                                                                                                                                                                                                                                                                              | `python3 dev/write_sm120_round_manifest.py ...` passed; build-only harness validation passed with `--require-manifest`.                                                                                                                                                                                                                                |
| 2026-05-20 | Runtime correctness          | Promoted the runtime-family smokes into the SM120 round correctness gate: bias add/reduction, GELU, fused classifier/dlogits, encoder, AdamW, and global norm now build, run, and validate alongside matmul/attention/LayerNorm.                                                                                                                                                                                                                                                                | `make -j test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` passed. Fresh runtime logs should be captured by the next full SM120 round.                                                                                                                                  |
| 2026-05-20 | Runtime coverage             | Added `bench_sm120_runtime` for non-GEMM GPT-2 kernels: bias add/reduction, GELU, fused classifier, AdamW, global norm, encoder, memsets, and copies.                                                                                                                                                                                                                                                                                                                                           | `make -j bench_sm120_runtime DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` passed. Fresh runtime timings should be captured by the next full SM120 round.                                                                                                                                                                                              |
| 2026-05-20 | Accumulated dWeight coverage | Added `dW+accum` rows to `bench_sm120_matmul` for the gradient-accumulation path used by the trainer.                                                                                                                                                                                                                                                                                                                                                                                           | `make -j bench_sm120_matmul DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` passed. Fresh runtime timings should be captured by the next full SM120 round.                                                                                                                                                                                              |
| 2026-05-20 | Optional stack feasibility   | Added `dev/probe_sm120_backend_stacks.py` and wired the round harness to record ThunderKittens, Plain CUDA, GPU runtime, cuBLAS, cuBLASLt, cuDNN, Triton, and CuTeDSL availability before candidate selection.                                                                                                                                                                                                                                                                                  | `python3 dev/probe_sm120_backend_stacks.py --json-out /tmp/sm120_backend_stacks.json --markdown-out /tmp/sm120_backend_stacks.md` passed.                                                                                                                                                                                                              |
| 2026-05-20 | Fused dGELU coverage         | Added an MLP-projection backward `dInp+dGeLU` row to `bench_sm120_matmul`, comparing TK fused dGELU, cuBLASLt DGELU epilogue, and cuBLASLt plus explicit `gelu_backward_inplace`.                                                                                                                                                                                                                                                                                                               | `make -j bench_sm120_matmul DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` passed. Fresh runtime timings should be captured by the next full SM120 round.                                                                                                                                                                                              |
| 2026-05-20 | Artifact parsing             | Added `dev/validate_sm120_round.py` and wired the round harness to validate required logs, parse benchmark/training metrics, check checkpoint cleanup, and emit `scoreboard-candidates.md`.                                                                                                                                                                                                                                                                                                     | `python3 dev/validate_sm120_round.py --self-test` and build-only harness validation passed.                                                                                                                                                                                                                                                            |
| 2026-05-20 | Round harness                | Added `scripts/run_sm120_optimization_round.sh` to run the SM120 build, correctness gates, microbenchmarks, TinyStories smoke, summary capture, and safe checkpoint cleanup as one repeatable protocol.                                                                                                                                                                                                                                                                                         | `bash -n` passed; `DRY_RUN=1` printed the expected commands; build-only mode compiled all targets.                                                                                                                                                                                                                                                     |
| 2026-05-20 | cuBLASLt selector            | Repaired the worktree away from a reintroduced max-waves default. The source default is back to 8 heuristic results with lowest-wave selection unless an explicit index or max-waves override is passed.                                                                                                                                                                                                                                                                                        | Existing changelog evidence rejected `LLMK_SM120_CUBLASLT_SELECT_MAX_WAVES=1` at `3319.89 ms`; compile gate passed after repair.                                                                                                                                                                                                                       |
| 2026-05-20 | Microbench coverage          | Added repeatable `bench_sm120_attention` and `bench_sm120_layernorm` Makefile targets; LayerNorm timing now includes plain forward, fused residual+LayerNorm forward, and backward.                                                                                                                                                                                                                                                                                                             | `make -j bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1` passed. Fresh runtime timings should be captured by the next full SM120 round.                                                                                                                                                 |


## Training Runs


| Run                                 | Config                                                          | Steps         | Avg Step Time                                                         | Throughput                      | Decision                                                                              |
| ----------------------------------- | --------------------------------------------------------------- | ------------- | --------------------------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------- |
| llm.c baseline from `new-goal.md`   | User-supplied GPT-2 TinyStories command                         | 3             | about `2818.23 ms` including step 1, about `2681.78 ms` for steps 2-3 | about `195k tok/s` steady state | Target to beat.                                                                       |
| Restored SM120 baseline             | `SM120_USE_CUBLASLT_GEMM=1`, no master weights, `gelu_fusion=1` | 3             | `2508.27 ms`                                                          | `208.9k` to `209.2k tok/s`      | Current documented baseline in `docs/sm120-rtx5090-baseline.md`.                      |
| Current best noted in `new-goal.md` | Same TinyStories command, 10-step run                           | first 3 of 10 | about `2469.84 ms`                                                    | about `212k tok/s`              | Historical note only; current same-harness x10 evidence did not reproduce it. |
| `codex_sm120_round_backward_stream_sync_default_x10_20260521` | Promoted default SM120 `main_stream` synchronization after backward instead of device-wide sync | 10 | `2493.133 ms` | `210.2k` to `211.1k tok/s` | Current stable same-harness default; x10 improvement over `2495.443 ms`, with full native correctness, benchmark, manifest, stack-probe, training, and audit evidence. |
| `codex_sm120_round_backward_stream_sync_x10_20260521` | Opt-in SM120 `main_stream` synchronization after backward instead of device-wide sync | 10 | `2495.290 ms` | `210.0k` to `210.9k tok/s` | Superseded by the default-build x10 row above; useful as the pre-promotion equivalent-macro stability evidence. |
| `codex_sm120_round_backward_stream_sync_20260521` | Opt-in SM120 `main_stream` synchronization after backward instead of device-wide sync | 3 | `2483.598 ms` | `210.7k` to `211.3k tok/s` | Promising 3-step candidate that justified x10 validation; superseded by the x10 row above. |
| `codex_sm120_round_cublas_dinp_attproj_fc_20260521` | Candidate direct-cuBLAS dInput for GPT-2 attproj and MLP-up backward | 3 | `2493.931 ms` | `209.8k` to `210.3k tok/s` | 3-step win, but rejected after the x10 stability round regressed; source restored to the prior selector. |
| `codex_sm120_round_cublas_dinp_attproj_fc_x10_20260521` | Candidate direct-cuBLAS dInput for GPT-2 attproj and MLP-up backward | 10 | `2502.950 ms` | `209.4k` to `210.3k tok/s` | Rejected; slower than current stable x10 default `2495.443 ms`, despite passing all validation gates. |
| `codex_sm120_round_cublas_dinp_attproj_only_20260521` | Candidate direct-cuBLAS dInput for GPT-2 attproj backward only | 3 | `2491.581 ms` | `210.0k` to `210.5k tok/s` | Promising focused 3-step selector that required x10 validation before any promotion. |
| `codex_sm120_round_cublas_dinp_attproj_only_x10_20260521` | Candidate direct-cuBLAS dInput for GPT-2 attproj backward only | 10 | `2503.023 ms` | `209.4k` to `210.5k tok/s` | Rejected; slower than current native x10 default `2493.133 ms`, despite passing all validation gates. |
| `codex_sm120_round_current_mix_final_smoke_20260521` | Full SM120 round with current mixed backend after Torch/Triton/cuDNN/CuTeDSL comparison work | 3 | `2494.781 ms` | `209.7k` to `210.3k tok/s` | Current 3-step smoke evidence: correctness, benchmarks, stack probe, manifest, checkpoint cleanup, and TinyStories training all validated. Stable-best claim remains the x10 default row below. |
| `codex_sm120_round_current_native_x10_median_20260521` | Current-source native x10 round with median attention benchmark and 93-row native benchmark contract | 10 | `2496.245 ms` | `210.0k` to `210.8k tok/s` | Validated current-source evidence and now drives `current-sm120-selection`; not a new stable-best claim versus `2495.443 ms`. Selection filters 45 native selected benchmark rows down to 42 GPT-2 objective trainer rows and records three C=3072 LayerNorm rows as extra benchmark evidence. |
| `codex_sm120_round_current_default_x10_after_memory_20260520` | Full SM120 round with current mixed backend and expanded memory-shape contract | 10 | `2495.443 ms` | `210.1k` to `210.8k tok/s` | Superseded by `codex_sm120_round_backward_stream_sync_default_x10_20260521`; still useful as the prior stable x10 comparison point. |
| `codex_sm120_round_classifier_pad_tail_zero_20260520` | Candidate classifier dlogits padded-tail zeroing | 3 | `2596.574 ms` | `201.7k` to `202.0k tok/s` | Rejected and source restored; stronger standalone padding contract regressed versus current-default x10 `2495.443 ms`. |
| `codex_sm120_round_layernorm_smem_attr_cache_x10_20260520` | Candidate cached LayerNorm dynamic-smem attribute setup | 10 | `2630.194 ms` | `198.9k` to `201.0k tok/s` | Rejected and source restored; slower than current-default x10 `2495.443 ms` despite passing all validation gates. |
| `codex_sm120_round_layernorm_smem_attr_cache_20260520` | Candidate cached LayerNorm dynamic-smem attribute setup | 3 | `2494.965 ms` | `209.9k` to `210.2k tok/s` | Correct and validated, but not promoted after the x10 stability round regressed. |
| `codex_sm120_round_cublas_bwd_default_x10_20260520` | Full SM120 round with default direct-cuBLAS backward selector | 10 | `2497.246 ms` | `209.9k` to `210.7k tok/s` | Superseded by `codex_sm120_round_current_default_x10_after_memory_20260520`; still useful as the prior stable comparison point. |
| `codex_sm120_round_memory_shape_coverage_20260520` | Full SM120 round with logits and hidden memory-shape coverage required by the validator | 3 | `2490.206 ms` | `210.2k` to `210.9k tok/s` | Validated coverage/contract round on RTX 5090; useful evidence for memory-overhead coverage, not a speed promotion versus current-default x10 `2497.246 ms`. |
| `codex_sm120_round_runtime_shape_validator_20260520` | Full SM120 round with runtime shape validator, bias-add vec2, and expanded runtime rows | 3 | `2545.384 ms` | `205.9k` to `209.1k tok/s` | Validated coverage/contract round; not a speed promotion versus current-default x10 `2497.246 ms` or the fresh 3-step default `2495.261 ms`. |
| `codex_sm120_round_classifier_loss_only_x10_20260520` | Full SM120 round with classifier loss-only specialization | 10 | `2551.284 ms` | `201.9k` to `206.7k tok/s` | Correct and validated, but not a stable training-speed promotion versus the current-default x10 run. |
| `codex_sm120_round_classifier_loss_only_20260520` | Full SM120 round with classifier loss-only specialization | 3 | `2488.125 ms` | `209.2k` to `210.8k tok/s` | 3-step win and validation-loss micro-op accepted; needs stable x10 speed before any training-best claim. |
| `codex_sm120_round_cublas_bwd_default_rerun_20260520` | Full SM120 round with default direct-cuBLAS backward selector | 3 | `2495.261 ms` | `209.7k` to `210.2k tok/s` | Current fresh promoted default; faster than restored baseline and classifier round, still slower than historical `2469 ms` note. |
| `codex_sm120_round_cublas_lmhead_forward_20260520` | Opt-in direct-cuBLAS LM-head forward selector on top of current default | 3 | `2522.413 ms` | `207.2k` to `208.6k tok/s` | Rejected and reverted; focused LM-head cuBLAS timing was lower, but trainer integration regressed versus the current `2495-2497 ms` default range. |
| `codex_sm120_round_cublas_bwd_20260520` | Same selector enabled with `EXTRA_NVCC_FLAGS=-DLLMK_SM120_USE_CUBLAS_BACKWARD_GEMM` before default promotion | 3 | `2493.738 ms` | `209.8k` to `210.4k tok/s` | Confirms the selector can improve end-to-end; paired with default rerun because one default round was noisy. |
| `codex_sm120_round_cublas_bwd_default_20260520` | First default validation round for the direct-cuBLAS backward selector | 3 | `2542.302 ms` | `203.3k` to `206.3k tok/s` | Passed validation but treated as noisy/rejected evidence because the immediate rerun matched the opt-in win. |
| `codex_sm120_round_classifier_1024_20260520` | Full SM120 round harness with SM120 two-pass 1024-thread classifier | 3 | `2528.193 ms` | `207.0k` to `207.4k tok/s` | Faster than the prior validated `2577.055 ms` round; still rejected as a speed promotion versus the restored baseline. |
| `codex_sm120_round_adamw1024_20260520` | Full SM120 round harness with AdamW 1024-thread/no-master launch experiment | 3 | `2567.526 ms` | `202.5k` to `204.3k tok/s` | Rejected and reverted; slower than the retained classifier round and not a speed promotion. |
| `codex_sm120_round_after_lnfix_clsfix_20260520` | Full SM120 round harness, manifest + stack probe + all correctness and benchmark gates | 3 | `2577.055 ms` | `201.5k` to `203.5k tok/s` | Validated correctness/coverage after fixes; rejected as a speed promotion versus the restored baseline. |
| `log124M/5090_S_attn_16.log`        | B=64, `gelu_fusion=1`, no master weights, `-x 5`                | 5             | `2521.28 ms`                                                          | `207.5k` to `208.5k tok/s`      | Recent local evidence; slower than the historical `2469 ms` note.                     |
| `log124M/5090_S_attn_32.log`        | B=64, `gelu_fusion=1`, no master weights, `-x 5`                | 5             | `2535.37 ms`                                                          | `205.7k` to `207.1k tok/s`      | Rejected versus `attn_16` and the documented baseline.                                |


## Matmul - GEMM

### Forward `A * B^T`


| Kernel               | Shape `(M, N, K)`   | Stack    | Config           | Time (ms) | Decision                                                              |
| -------------------- | ------------------- | -------- | ---------------- | --------- | --------------------------------------------------------------------- |
| qkv                  | `65536, 2304, 768`  | cuBLASLt | default fallback | `1.039`   | Keep cuBLASLt; opt-in TK trainer round regressed to `2598.442 ms`.     |
| qkv                  | `65536, 2304, 768`  | TK       | rejected opt-in  | `1.073`   | Rejected; round-local cuBLASLt was faster and TinyStories regressed.   |
| qkv                  | `65536, 2304, 768`  | Torch    | prototype        | `1.449`   | Rejected; slower than same-session cuBLASLt `1.046 ms`.                |
| qkv                  | `65536, 2304, 768`  | Triton   | prototype        | `2.263`   | Rejected; slower than cuBLASLt, TK, and Torch.                         |
| attention projection | `65536, 768, 768`   | cuBLASLt | default fallback | `0.372`   | Keep cuBLASLt; TK median was `0.377 ms`.                              |
| attention projection | `65536, 768, 768`   | Torch    | prototype        | `0.523`   | Rejected; slower than same-session cuBLASLt `0.370 ms`.               |
| attention projection | `65536, 768, 768`   | Triton   | prototype        | `0.735`   | Rejected; slower than cuBLASLt, TK, and Torch.                         |
| MLP up               | `65536, 3072, 768`  | cuBLASLt | fused GELU       | `1.470`   | Keep cuBLASLt fused GELU; TK fused median was `1.581 ms`.              |
| MLP up               | `65536, 3072, 768`  | Torch    | fwd+GELU prototype | `2.441` | Rejected; much slower than cuBLASLt fused GELU.                       |
| MLP up               | `65536, 3072, 768`  | Triton   | fwd+GELU prototype | `3.169` | Rejected; slower than cuBLASLt, TK, and Torch.                        |
| MLP projection       | `65536, 768, 3072`  | cuBLASLt | default fallback | `1.368`   | Keep cuBLASLt; repeated-sample evidence rejected the older TK row.     |
| MLP projection       | `65536, 768, 3072`  | Torch    | prototype        | `1.584`   | Rejected; slower than same-session cuBLASLt `1.402 ms`.               |
| MLP projection       | `65536, 768, 3072`  | Triton   | prototype        | `3.285`   | Rejected; slower than cuBLASLt, TK, and Torch.                         |
| LM-head              | `65536, 50304, 768` | cuBLAS   | rejected opt-in  | `22.186`  | Focused row beat cuBLASLt, but the opt-in trainer round regressed to `2522.413 ms`. |
| LM-head              | `65536, 50304, 768` | cuBLASLt | default fallback | `23.528`  | Keep cuBLASLt; end-to-end trainer timing beats the direct-cuBLAS opt-in route. |
| LM-head              | `65536, 50304, 768` | Torch    | prototype        | `24.324`  | Rejected; slower than same-session cuBLAS/cuBLASLt rows.              |
| LM-head              | `65536, 50304, 768` | Triton   | prototype        | `49.373`  | Rejected; much slower than cuBLAS, cuBLASLt, and Torch.                |


### dInput `A * B`


| Shape `(M, N, K)`   | Stack    | Config               | Time (ms) | Decision                                                                           |
| ------------------- | -------- | -------------------- | --------- | ---------------------------------------------------------------------------------- |
| `65536, 2304, 768`  | cuBLASLt | default fallback     | `1.099`   | Keep cuBLASLt.                                                                     |
| `65536, 2304, 768`  | TK       | `grad_128x64`        | `1.151`   | Rejected.                                                                          |
| `65536, 2304, 768`  | Torch    | prototype            | `1.020`   | Marginal same-session row, but no promotion without a trainer-callable Torch route. |
| `65536, 2304, 768`  | Triton   | prototype            | `2.504`   | Rejected; slower than cuBLAS/cuBLASLt/Torch.                                       |
| `65536, 768, 768`   | cuBLASLt | default fallback     | `0.397`   | Keep cuBLASLt.                                                                     |
| `65536, 768, 768`   | TK       | `grad_128x64`        | `0.408`   | Rejected.                                                                          |
| `65536, 768, 768`   | Torch    | prototype            | `0.371`   | Rejected; same-session direct cuBLAS was faster at `0.366 ms`.                     |
| `65536, 768, 768`   | Triton   | prototype            | `0.721`   | Rejected; slower than cuBLAS/cuBLASLt/Torch.                                       |
| `65536, 3072, 768`  | cuBLASLt | default fallback     | `1.393`   | Keep cuBLASLt.                                                                     |
| `65536, 3072, 768`  | TK       | `grad_128x64`        | `1.474`   | Rejected.                                                                          |
| `65536, 3072, 768`  | Torch    | prototype            | `1.357`   | Rejected after focused refresh; current C++ cuBLAS was faster at `1.331 ms`.        |
| `65536, 3072, 768`  | Triton   | prototype            | `3.299`   | Rejected; slower than cuBLASLt/cuBLAS/Torch.                                       |
| `65536, 768, 3072`  | cuBLASLt | default fallback     | `1.438`   | Keep cuBLASLt.                                                                     |
| `65536, 768, 3072`  | TK       | `grad_128x64`        | `1.469`   | Rejected.                                                                          |
| `65536, 768, 3072`  | Torch    | prototype            | `1.397`   | Marginal same-session row versus cuBLAS/cuBLASLt; not a trainer promotion.         |
| `65536, 768, 3072`  | Triton   | prototype            | `2.969`   | Rejected; slower than cuBLASLt/cuBLAS/Torch.                                       |
| `65536, 3072, 768`  | TK       | exact fused dGELU opt-in | `1.762` | Correct and faster in focused timing, but rejected as default after x10 TinyStories regressed to `2502.328 ms`. |
| `65536, 3072, 768`  | cuBLASLt | fused dGELU epilogue | `1.855`   | Current trainer route; keep after the TK exact dGELU x10 rejection.                |
| `65536, 3072, 768`  | cuBLASLt | explicit dGELU pass  | `2.184`   | Rejected versus fused epilogue.                                                    |
| `65536, 3072, 768`  | Torch    | explicit dGELU prototype | `28.754` | Rejected; composed Torch dGELU is far slower than fused/explicit native rows. |
| `65536, 3072, 768`  | Triton   | dInp+dGeLU prototype | `3.619`   | Rejected; correct but slower than fused TK/cuBLASLt rows.                          |
| `65536, 50304, 768` | cuBLAS   | default large-OC route | `21.232` | Promoted for non-fused large-OC dInput; default rerun beat cuBLASLt `21.757 ms`.   |
| `65536, 50304, 768` | cuBLASLt | default fallback     | `21.757`  | Replaced by the direct-cuBLAS large-OC route.                                      |
| `65536, 50304, 768` | TK       | `grad_128x64`        | `23.761`  | Rejected.                                                                          |
| `65536, 50304, 768` | Torch    | prototype            | `21.746`  | Rejected; slower than same-session direct cuBLAS `21.433 ms`.                     |
| `65536, 50304, 768` | Triton   | prototype            | `52.580`  | Rejected; much slower than cuBLAS/cuBLASLt/Torch.                                  |


### dWeight `A^T * B`


| Shape `(M, N, K)`   | Stack    | Config           | Time (ms) | Decision       |
| ------------------- | -------- | ---------------- | --------- | -------------- |
| `2304, 768, 65536`  | cuBLAS   | default selector | `1.006`   | Promoted over cuBLASLt for qkv dWeight. |
| `2304, 768, 65536`  | cuBLASLt | default fallback | `1.106`   | Replaced by direct cuBLAS for this shape. |
| `2304, 768, 65536`  | TK       | `grad_128x64`    | `1.578`   | Rejected.      |
| `2304, 768, 65536`  | Torch    | prototype        | `1.005`   | Tie-range with promoted cuBLAS; no trainer promotion. |
| `2304, 768, 65536`  | Triton   | prototype        | `2.304`   | Rejected; slower than cuBLAS/cuBLASLt/Torch. |
| `768, 768, 65536`   | cuBLAS   | default selector | `0.358`   | Promoted over cuBLASLt for attention-projection dWeight. |
| `768, 768, 65536`   | cuBLASLt | default fallback | `0.396`   | Replaced by direct cuBLAS for this shape. |
| `768, 768, 65536`   | TK       | `grad_128x64`    | `0.579`   | Rejected.      |
| `768, 768, 65536`   | Torch    | prototype        | `0.338`   | Rejected; same-session direct cuBLAS was faster at `0.328 ms`. |
| `768, 768, 65536`   | Triton   | prototype        | `0.578`   | Rejected; slower than cuBLAS/cuBLASLt/Torch. |
| `3072, 768, 65536`  | cuBLAS   | default selector | `1.355`   | Promoted over cuBLASLt for MLP-up dWeight. |
| `3072, 768, 65536`  | cuBLASLt | default fallback | `1.475`   | Replaced by direct cuBLAS for this shape. |
| `3072, 768, 65536`  | TK       | `grad_128x64`    | `1.700`   | Rejected.      |
| `3072, 768, 65536`  | Torch    | prototype        | `1.347`   | Rejected after focused refresh; current C++ cuBLAS was faster at `1.333 ms`. |
| `3072, 768, 65536`  | Triton   | prototype        | `2.471`   | Rejected; slower than cuBLAS/cuBLASLt/Torch. |
| `768, 3072, 65536`  | cuBLAS   | default selector | `1.354`   | Promoted over cuBLASLt for MLP-projection dWeight. |
| `768, 3072, 65536`  | cuBLASLt | default fallback | `1.522`   | Replaced by direct cuBLAS for this shape. |
| `768, 3072, 65536`  | TK       | `grad_128x64`    | `1.766`   | Rejected.      |
| `768, 3072, 65536`  | Torch    | prototype        | `1.376`   | Rejected; same-session direct cuBLAS was faster at `1.357 ms`. |
| `768, 3072, 65536`  | Triton   | prototype        | `2.344`   | Rejected; slower than cuBLAS/cuBLASLt/Torch. |
| `50304, 768, 65536` | cuBLASLt | default fallback | `20.855`  | Keep cuBLASLt; direct cuBLAS was not stable enough for LM-head dWeight. |
| `50304, 768, 65536` | TK       | `grad_128x64`    | `26.023`  | Rejected.      |
| `50304, 768, 65536` | Torch    | prototype        | `21.517`  | Rejected; slower than same-session cuBLASLt `21.014 ms`. |
| `50304, 768, 65536` | Triton   | prototype        | `75.084`  | Rejected; much slower than cuBLASLt/Torch. |


### dWeight Accumulated `A^T * B + C`


| Shape `(M, N, K)`   | Stack    | Config                 | Time (ms) | Decision                                                                           |
| ------------------- | -------- | ---------------------- | --------- | ---------------------------------------------------------------------------------- |
| `2304, 768, 65536`  | cuBLAS   | default selector       | `0.990`   | Promoted over cuBLASLt for accumulated qkv dWeight.                                |
| `2304, 768, 65536`  | cuBLASLt | accumulate             | `1.126`   | Replaced by direct cuBLAS for this shape.                                          |
| `2304, 768, 65536`  | TK       | split-K or scratch add | `1.589`   | Rejected in the validated round.                                                   |
| `2304, 768, 65536`  | Torch    | prototype              | `1.008`   | Rejected; slower than same-session direct cuBLAS `0.998 ms`.                       |
| `2304, 768, 65536`  | Triton   | prototype              | `2.316`   | Rejected; slower than cuBLAS/cuBLASLt/Torch.                                       |
| `768, 768, 65536`   | cuBLAS   | default selector       | `0.329`   | Promoted over cuBLASLt for accumulated attention-projection dWeight.               |
| `768, 768, 65536`   | cuBLASLt | accumulate             | `0.410`   | Replaced by direct cuBLAS for this shape.                                          |
| `768, 768, 65536`   | TK       | split-K or scratch add | `0.601`   | Rejected in the validated round.                                                   |
| `768, 768, 65536`   | Torch    | prototype              | `0.344`   | Rejected; slower than same-session direct cuBLAS `0.333 ms`.                       |
| `768, 768, 65536`   | Triton   | prototype              | `0.586`   | Rejected; slower than cuBLAS/cuBLASLt/Torch.                                       |
| `3072, 768, 65536`  | cuBLAS   | default selector       | `1.311`   | Promoted over cuBLASLt for accumulated MLP-up dWeight.                             |
| `3072, 768, 65536`  | cuBLASLt | accumulate             | `1.541`   | Replaced by direct cuBLAS for this shape.                                          |
| `3072, 768, 65536`  | TK       | split-K or scratch add | `1.867`   | Rejected in the validated round.                                                   |
| `3072, 768, 65536`  | Torch    | prototype              | `1.352`   | Tie-range with same-session direct cuBLAS `1.354 ms`; no trainer promotion.        |
| `3072, 768, 65536`  | Triton   | prototype              | `2.476`   | Rejected; slower than cuBLAS/cuBLASLt/Torch.                                       |
| `768, 3072, 65536`  | cuBLAS   | default selector       | `1.328`   | Promoted over cuBLASLt for accumulated MLP-projection dWeight.                     |
| `768, 3072, 65536`  | cuBLASLt | accumulate             | `1.465`   | Replaced by direct cuBLAS for this shape.                                          |
| `768, 3072, 65536`  | TK       | split-K or scratch add | `1.710`   | Rejected in the validated round.                                                   |
| `768, 3072, 65536`  | Torch    | prototype              | `1.399`   | Rejected; same-session direct cuBLAS was faster at `1.320 ms`.                     |
| `768, 3072, 65536`  | Triton   | prototype              | `2.347`   | Rejected; slower than cuBLAS/cuBLASLt/Torch.                                       |
| `50304, 768, 65536` | cuBLASLt | accumulate             | `20.983`  | Keep cuBLASLt; direct cuBLAS was not stable enough for LM-head dWeight.            |
| `50304, 768, 65536` | TK       | split-K or scratch add | `26.373`  | Rejected in the validated round.                                                   |
| `50304, 768, 65536` | Torch    | prototype              | `21.522`  | Rejected; slower than same-session cuBLASLt `21.011 ms`.                           |
| `50304, 768, 65536` | Triton   | prototype              | `76.736`  | Rejected; much slower than cuBLASLt/Torch.                                         |


## Attention - MHA


| Shape `(B, NH, T, HS)` | Pass     | Stack | Config     | Time (ms) | Decision            |
| ---------------------- | -------- | ----- | ---------- | --------- | ------------------- |
| `64, 12, 1024, 64`     | forward  | TK    | packed-QKV | `0.779`   | Keep TK packed-QKV for the trainer. |
| `64, 12, 1024, 64`     | backward | TK    | packed-QKV | `2.729`   | Keep TK packed-QKV for the trainer. |
| `64, 12, 1024, 64`     | forward  | Torch | native SDPA, separated Q/K/V | `0.555` | Use for Python-side separated-Q/K/V reference rows; not a trainer route. |
| `64, 12, 1024, 64`     | backward | Torch | native SDPA, separated Q/K/V | `2.196` | Use for Python-side separated-Q/K/V reference rows; not a trainer route. |
| `64, 12, 1024, 64`     | forward  | TorchPacked | packed-QKV strided prototype | `1.068` | Rejected for trainer integration; slower than same-session packed TK. |
| `64, 12, 1024, 64`     | backward | TorchPacked | packed-QKV strided prototype | `4.068` | Rejected for trainer integration; slower than same-session packed TK. |
| `64, 12, 1024, 64`     | forward  | TorchMaterializedPacked | packed QKV with explicit Q/K/V materialization | `1.261` | Rejected; materializing Q/K/V is slower than the strided packed Torch route and packed TK. |
| `64, 12, 1024, 64`     | backward | TorchMaterializedPacked | packed QKV with explicit Q/K/V materialization | `4.196` | Rejected; materialization does not make Torch trainer-layout backward competitive. |
| `64, 12, 1024, 64`     | forward  | cuDNN | native SDPA, separated Q/K/V | `0.686` | Correct feasibility row; slower than native Torch SDPA and not a trainer route. |
| `64, 12, 1024, 64`     | backward | cuDNN | native SDPA, separated Q/K/V | `2.376` | Correct feasibility row; slower than native Torch SDPA and not a trainer route. |
| `64, 12, 1024, 64`     | forward  | cuDNNPacked | packed-QKV prototype | `0.803` | Does not beat packed TK by enough across forward plus backward to justify a cuDNN trainer link. |
| `64, 12, 1024, 64`     | backward | cuDNNPacked | packed-QKV prototype | `3.514` | Rejected for trainer integration; slower than packed TK backward. |
| `64, 12, 1024, 64`     | forward  | Triton | separated Q/K/V prototype | `2.075` | Rejected; correct forward-only row, but slower than Torch, cuDNN, and packed TK. |
| `64, 12, 1024, 64`     | forward  | TritonPacked | packed-QKV prototype | `2.197` | Rejected; correct trainer-layout forward-only row, but slower than packed TK, TorchPacked, and cuDNNPacked. |


## LayerNorm


| Width  | Pass                   | Stack | Config   | Time (ms) | Decision                                                                              |
| ------ | ---------------------- | ----- | -------- | --------- | ------------------------------------------------------------------------------------- |
| `768`  | forward                | CUDA  | baseline | `0.142`   | Keep CUDA shared-memory baseline; no-smem forward regressed TinyStories.              |
| `768`  | forward                | Triton | rejected prototype | `0.178` | Correct against Torch FP32 reference but slower than CUDA; do not wire into trainer. |
| `768`  | forward                | Torch native | rejected prototype | `0.153` | Close to CUDA but does not expose saved mean/rstd needed by the trainer. |
| `768`  | forward                | Torch with stats | rejected prototype | `2.223` | Correct and trainer-state-compatible, but much slower than CUDA. |
| `768`  | fused residual forward | CUDA  | baseline | `0.283`   | Keep fused CUDA baseline; no-smem fallback regressed focused timing.                  |
| `768`  | fused residual forward | Triton | rejected prototype | `0.311` | Correct but slower than CUDA. |
| `768`  | fused residual forward | Torch native | rejected prototype | `0.339` | Slower than CUDA and does not expose saved mean/rstd. |
| `768`  | fused residual forward | Torch with stats | rejected prototype | `3.226` | Correct and trainer-state-compatible, but much slower than CUDA. |
| `768`  | backward               | CUDA  | baseline | `0.290`   | Keep CUDA baseline; accumulator fix is validated on SM120.                            |
| `768`  | backward               | Triton atomic FP32-grad | rejected prototype | `0.364` | Full gradients are correct, but slower than CUDA and dweight/dbias use an FP32 buffer contract. |
| `768`  | backward               | Torch native | rejected prototype | `0.416` | Full saved-mean/rstd backward is correct but slower than CUDA; do not add libtorch. |
| `768`  | backward dInput + BF16 dweight/dbias | Torch native + reductions | rejected hybrid | `1.964` | Closes the dInput-only promotion gate: adding Torch reductions for trainer-required BF16 dweight/dbias is much slower than CUDA full backward. |
| `768`  | backward dInput only   | Torch native | resolved partial | `0.222` | Fastest dInput-only row, but not trainer-equivalent; the full hybrid with BF16 dweight/dbias is rejected. |
| `768`  | backward dInput only   | Triton | resolved partial | `0.228` | Correct dInput decomposition row; slower than Torch dInput-only and not trainer-equivalent alone. |
| `3072` | forward                | CUDA  | baseline | `0.542`   | Keep CUDA baseline until a fused/Triton/CuTeDSL candidate wins end-to-end.            |
| `3072` | forward                | Triton | rejected prototype | `0.581` | Correct against Torch FP32 reference but slower than CUDA; do not wire into trainer. |
| `3072` | forward                | Torch native | rejected prototype | `0.554` | Close to CUDA but does not expose saved mean/rstd needed by the trainer. |
| `3072` | forward                | Torch with stats | rejected prototype | `9.094` | Correct and trainer-state-compatible, but much slower than CUDA. |
| `3072` | fused residual forward | Triton | rejected prototype | `1.121` | Correct prototype row; no current CUDA fused-residual comparison row at this width. |
| `3072` | fused residual forward | Torch native | rejected prototype | `1.335` | Correct but no saved mean/rstd and slower than Triton prototype. |
| `3072` | fused residual forward | Torch with stats | rejected prototype | `13.346` | Correct and trainer-state-compatible, but much slower than CUDA-style expectations. |
| `3072` | backward               | CUDA  | baseline | `1.105`   | Keep CUDA baseline until a fused/Triton/CuTeDSL candidate wins end-to-end.            |
| `3072` | backward               | Torch native | non-trainer shape | `1.395` | Full backward is correct but slower than CUDA, and this width is not GPT-2 LayerNorm. |
| `3072` | backward               | Triton atomic FP32-grad | non-trainer shape | `1.423` | Full gradients are correct, but slower than CUDA and not a GPT-2 LayerNorm target. |
| `3072` | backward dInput only   | Triton | non-trainer partial | `0.834` | Fastest dInput-only row at this stress width; not a GPT-2 trainer LayerNorm target. |
| `3072` | backward dInput only   | Torch native | non-trainer partial | `0.838` | Correct dInput-only row; slower than Triton and missing dweight/dbias. |


## Runtime / Pointwise / Reduction

The `bench_sm120_runtime` target covers these plain CUDA or CUDA-runtime
baselines. Timings below cite the current stable x10 round
`scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520`.


| Kernel Family                  | Shape                                       | Stack        | Time    | Decision                                                              |
| ------------------------------ | ------------------------------------------- | ------------ | ------- | --------------------------------------------------------------------- |
| Bias add                       | `BT=65536, OC=768 / 3072`                   | CUDA         | `82.143 us / 554.136 us` | Promoted SM120 vec2 add for aligned standalone rows; scalar fallback remains available for unaligned widths. |
| Bias add                       | `BT=65536, OC=768 / 3072`                   | Triton       | `146.328 us / 573.555 us` | Rejected; slower than same-machine CUDA. |
| Bias gradient reduction        | `BT=65536, OC=768 / 2304 / 3072`            | CUDA         | `24.464 us / 187.960 us / 247.230 us` | Baseline provider; benchmark now covers the trainer-active GPT-2 bias widths. |
| Bias gradient reduction        | `BT=65536, OC=768 / 2304 / 3072`            | Torch        | `325.437 us / 1018.461 us / 1358.525 us` | Rejected; slower than CUDA. |
| GELU forward/backward          | `BT=65536, C=3072`                          | CUDA         | `528.159 us / 781.004 us` | Baseline provider unless fused epilogue/dGELU improves trainer smoke. |
| GELU forward/backward          | `BT=65536, C=3072`                          | Triton       | `529.627 us / 792.446 us` | Rejected after focused refresh; correct, but slower than same-machine CUDA. |
| GELU forward/backward          | `BT=65536, C=3072`                          | Torch        | `547.383 us / 27327.625 us` | Rejected; backward composition is much slower than CUDA. |
| Fused classifier loss-only     | `B=64, T=1024, V=50257, P=50304`            | CUDA         | `3972.230 us` | Accepted for validation/forward-loss calls; logits remain untouched and the training dlogits path is unchanged. |
| Fused classifier loss-only     | `B=64, T=1024, V=50257, P=50304`            | Triton       | `8379.872 us` | Rejected; tiled Triton log-sum-exp is correct but slower than CUDA. |
| Fused classifier loss-only     | `B=64, T=1024, V=50257, P=50304`            | Torch        | `18131.519 us` | Rejected; BF16 Torch `cross_entropy` is much slower than CUDA. |
| Fused classifier / dlogits     | `B=64, T=1024, V=50257, P=50304`            | CUDA         | `8985.197 us` | Keep SM120 two-pass 1024-thread path; block sizes `256`, `512`, `768`, one-pass online softmax prep, target-logit cache, loss-sync elision, and exp2 softmax math were rejected. |
| Fused classifier / dlogits     | `B=64, T=1024, V=50257, P=50304`            | Triton       | `21981.632 us` | Rejected; full Triton dlogits is correct but much slower than CUDA. |
| Fused classifier / dlogits     | `B=64, T=1024, V=50257, P=50304`            | Torch        | `34081.570 us` | Rejected; autograd dlogits over the padded classifier shape is much slower than CUDA. |
| AdamW                          | `124475904` GPT-2 params, no master weights | CUDA         | `1856.336 us` | Baseline provider.                                                    |
| AdamW                          | `124475904` GPT-2 params, no master weights | Torch BF16-state | `1225.190 us` | Reference only; faster but not trainer-equivalent because Torch fused AdamW keeps BF16 moment buffers for BF16 params. |
| AdamW                          | `124475904` GPT-2 params, no master weights | Torch FP32-state | `7452.800 us` | Rejected; trainer-equivalent FP32 moment buffers are much slower than CUDA. |
| Global norm                    | `124475904` GPT-2 grads                     | CUDA         | `185.869 us` | Baseline provider.                                                    |
| Global norm                    | `124475904` GPT-2 grads                     | Torch        | `2366.618 us` | Rejected; slower than CUDA.                                           |
| Encoder forward                | `B=64, T=1024, C=768`                       | CUDA         | `81.052 us` | Baseline provider.                                                    |
| Encoder forward                | `B=64, T=1024, C=768`                       | Torch        | `203.051 us` | Rejected; slower than CUDA.                                           |
| Memset / device-to-device copy | `3296722944` BF16 padded-logits elements    | CUDA runtime | `4194.509 us / 8867.174 us` | Baseline classifier-sized activation overhead; now required in future runtime shape coverage. |
| Memset / device-to-device copy | `3296722944` BF16 padded-logits elements    | Torch        | `4397.152 us / 9310.240 us` | Rejected; slower than CUDA runtime. |
| Memset / device-to-device copy | `50331648` BF16 hidden elements             | CUDA runtime | `63.199 us / 133.575 us` | Baseline hidden-state runtime overhead; now required in future runtime shape coverage. |
| Memset / device-to-device copy | `50331648` BF16 hidden elements             | Torch        | `64.218 us / 134.214 us` | Tie-range, but no reason to replace direct CUDA runtime calls. |


## Current Mixed Strategy

- GEMM forward: use cuBLASLt for qkv, attention projection, MLP up, MLP projection, and LM-head. qkv TK and LM-head cuBLAS have small repeated-sample row wins, but neither is promoted without trainer evidence. Triton GEMM forward and fused-GeLU rows were benchmarked and are slower than the current cuBLAS/cuBLASLt/TK/Torch rows. The focused cuBLASLt fused-MLP epilogue algorithm probe found the current default heuristic index is already fastest for `fc` forward+GeLU.
- GEMM dInput: use cuBLASLt except non-fused large-OC rows, where direct cuBLAS is the default.
- GEMM dWeight: use direct cuBLAS for qkv, attention projection, MLP up, and MLP projection; keep LM-head dWeight on cuBLASLt. Triton dInput, dWeight, accumulated dWeight, and dInp+dGeLU rows are recorded as rejected comparison evidence. The focused cuBLASLt fused-MLP epilogue algorithm probe also found the current default heuristic index is fastest for `fcproj` dInput+dGeLU.
- Attention: use TK packed-QKV forward and backward in the trainer. Native
  Torch SDPA is faster for already-separated Q/K/V tensors and should remain
  the Python-side reference row, but the packed-QKV Torch paths are slower than
  TK after strided layout, explicit materialization, and packed-gradient handling.
  A full qkv-projection plus attention layout probe also rejects a separated
  Torch layout rewrite for the current trainer: split-strided Torch improves the
  backward side but loses more in forward, and the combined route remains slower
  than qkv cuBLASLt/cuBLAS plus packed TK.
  cuDNN SDPA is now benchmarked:
  separated cuDNN is correct but slower than native Torch SDPA, and the packed
  cuDNN path does not beat TK across forward plus backward enough to justify a
  trainer link. Triton separated and packed-QKV forward rows are now
  benchmarked and rejected as standalone forward-only rows.
- LayerNorm: use the CUDA baseline for trainer-equivalent forward, fused
  residual forward, and full backward. Torch and Triton dInput-only backward
  rows are useful decomposition evidence, but they are not trainer replacements.
  A focused Torch hybrid that adds BF16 dweight/dbias reductions to the native
  dInput-only row measured `1964.480 us` at `C=768`, so it is rejected.
- Classifier: use the SM120 two-pass 1024-thread dlogits path for training and
  the early-return loss-only path for validation/forward loss. Torch BF16
  `cross_entropy` and tiled Triton loss/dlogits prototypes were tested on the
  padded GPT-2 shape and rejected. The opt-in exp2 softmax math path also lost
  its same-session A/B against the default expf path.
- Runtime kernels: use the plain CUDA/CUDA-runtime baselines, with SM120 vec2
  bias add for aligned standalone rows. Trainer-active runtime replacements
  still need both a focused benchmark and TinyStories smoke evidence. Torch
  fused AdamW with BF16 moment state is a useful reference row, but it is not
  the trainer-equivalent FP32-moment route.
- Next highest-payoff work: move beyond the repeated GEMM row candidates that
  have failed promotion (`qkv` forward TK, LM-head forward cuBLAS, and TK exact
  fused dGELU), the noise-level attention dprep warp-count retest, and the
  rejected runtime launch sweeps (`bias_grad_reduce` block-size and grid-y caps,
  AdamW 1024-thread/no-master variants). Triton `3.6.0` and
  nvidia-cutlass-dsl `4.5.1` are available in the repo's `llm-kittens` conda
  env, and the harness now uses that interpreter when `CONDA_PREFIX` is active.
  Naive Triton GEMM, LayerNorm, separated and packed attention forward, and
  pointwise runtime prototypes are correct but mostly slower than
  CUDA/TK/Torch baselines. The useful follow-up should target a trainer-shaped
  backward-capable attention route or a native/codegen GEMM epilogue with a
  larger same-shape edge, not another standalone forward-only row or a
  cuBLASLt heuristic-index sweep of the existing fused MLP epilogues.
  Torch native SDPA is a useful separated-Q/K/V reference, but neither the
  strided nor materialized trainer-shaped packed path, nor a split-qkv
  separated layout route, justifies libtorch integration.
  cuDNN is available through the active Python `nvidia.cudnn` wheel path and
  has an opt-in attention feasibility row. The saved-forward packed cuDNN
  refresh fixes the earlier backward timing overcount, but the packed route
  still totals `3624.888 us`, behind packed TK's current `3507-3529 us`
  evidence. Do not add a cuDNN trainer link unless a refreshed packed-QKV
  forward/backward comparison beats TK by a material margin and survives the
  normal TinyStories smoke gate. Do not revisit rejected GEMM rows or
  runtime-only launch knobs without a new algorithmic hypothesis.
- Latest trainer composition check: `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1` plus
  `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` passed correctness but averaged
  `2498.079 ms` over three TinyStories steps, behind the `2493.133 ms` stable
  x10 baseline. This was a true combined-stack trial of two prior near-misses,
  not just an isolated scorecard row, and it was restored to the default
  no-candidate binary afterward.
- Latest GEMM/LayerNorm composition check: `LLMK_SM120_USE_CUBLAS_DINP_FC` plus
  `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` passed correctness but averaged
  `2515.866 ms` over three TinyStories steps. Combining the strongest previous
  FC-only dInput smoke with the LayerNorm candidate regressed badly, so the
  next useful work should avoid recombining rejected cuBLAS dInput selectors
  unless the selector implementation itself changes.
- Current runtime-state check: a fresh default x10 control averaged
  `2627.950 ms`, and the restored known default hash `dba87...` still averaged
  `2694.590 ms` over three direct steps. A concurrent dmon probe on the default
  hash averaged `2624.396 ms` while reporting 99-100% SM utilization, about
  574-577 W, 58-63 C, pclk 2707-2775 MHz, and no power or thermal violation
  flags. Treat this as a degraded runtime-state reference, not a new default
  promotion baseline; kernel candidates still need to beat the stable
  `2493.133 ms` x10 evidence before promotion.
- Live direct-script control: `./train-sm120.sh` on restored default binary
  hash `88b5510f...` averaged `2629.656 ms` over x10. The startup banner still
  reported `grad_zero_backend = CUDA runtime`,
  `dresidual_zero_backend = CUDA runtime`, `grad_scale_backend = host scalar`,
  and `gelu_fusion = 1`, so the current slowdown is not an active Torch route,
  LibTorch route, or leftover candidate flag. The diagnostic checkpoint files
  from this run were removed.
- LibTorch gradients-zero live retest: the real trainer `Torch C++`
  gradients-zero route passed all focused correctness smokes and averaged
  `2600.170 ms` over x3 in the current degraded runtime state, versus a
  restored default/no-candidate x3 control at `2615.825 ms`. This is a
  current-state relative win, but it is still much slower than the stable
  `2493.133 ms` x10 baseline, and the prior LibTorch grad-zero x10 round
  already regressed to `2495.623 ms`; keep it rejected for promotion. Default
  binaries were rebuilt afterward, and diagnostic checkpoint files were
  removed.
- LibTorch grad-zero plus precomputed AdamW grad-scale composition:
  `SM120_USE_LIBTORCH_GRAD_ZERO=1` with
  `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` passed all focused correctness
  smokes and averaged `2516.894 ms` over x3 with
  `grad_zero_backend = Torch C++` and
  `grad_scale_backend = precomputed device AdamW scalar`. The restored
  same-session default x3 averaged `2502.819 ms`, so the composition is slower
  by `14.075 ms` and remains behind the stable `2493.133 ms` x10 baseline.
  Keep both routes rejected for promotion; default binaries were restored and
  diagnostic checkpoints were removed.
- LibTorch grad-zero plus LayerNorm bwd1 composition:
  `SM120_USE_LIBTORCH_GRAD_ZERO=1` with
  `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` passed all focused correctness
  smokes and averaged `2549.664 ms` over x3 with
  `grad_zero_backend = Torch C++` and the default host scalar AdamW path. The
  restored same-session default x3 averaged `2597.681 ms`, so this is a
  current-state relative win of `48.017 ms`, but it is still `56.531 ms` slower
  than the stable `2493.133 ms` x10 baseline and does not rescue the LibTorch
  grad-zero route's prior x10 rejection. Default binaries were restored and
  diagnostic checkpoints were removed.
- LibTorch dresidual-zero plus wide bias-add composition:
  `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1` with
  `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024` passed all focused correctness
  smokes and averaged `2509.732 ms` over x3 with
  `dresidual_zero_backend = Torch C++`. The restored same-session default x3
  averaged `2511.081 ms`, so the candidate edge is only `1.349 ms`, which is
  noise-level, and the run remains `16.599 ms` slower than the stable
  `2493.133 ms` x10 baseline. Keep both component routes rejected for default
  promotion; default binaries were restored and diagnostic checkpoints were
  removed.
- Native wide bias-add plus LayerNorm bwd1 composition:
  `LLMK_SM120_BIAS_ADD_WIDE_BLOCK_SIZE=1024` with
  `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` passed all focused correctness
  smokes and averaged `2499.946 ms` over x3 on the native CUDA runtime paths.
  The restored same-session default x3 averaged `2503.331 ms`, so the
  candidate had a small `3.385 ms` current-session edge, but it is still
  `6.813 ms` slower than the stable `2493.133 ms` x10 baseline and both
  component routes have already failed x10 stability. Default binaries were
  restored and diagnostic checkpoints were removed.
- Stream-creation hypothesis: an opt-in
  `LLMK_SM120_NONBLOCKING_MAIN_STREAM` branch was added and tested. It passed
  focused smokes, but the trainer failed semantically with initial validation
  loss `0.002881`, final validation loss `0.000000`, and near-zero grad norms,
  while still averaging `2619.052 ms`. Keep the default blocking stream create
  path; the opt-in branch is recorded only to prevent retesting this unsafe
  synchronization shortcut.
- Grad-norm readback hypothesis: an opt-in
  `LLMK_SM120_ASYNC_GRAD_NORM_COPY` branch replaced the blocking scalar
  `cudaMemcpy` with `cudaMemcpyAsync` on `main_stream` plus
  `cudaStreamSynchronize(main_stream)`. Correctness and trainer losses/norms
  passed, but the x3 TinyStories average regressed to `2629.395 ms`; keep the
  original blocking readback path.
- Device grad-scale hypothesis: an opt-in
  `LLMK_SM120_DEVICE_GRAD_SCALE_ADAMW` branch computes the AdamW `grad_scale`
  from the device-resident grad-norm-squared scalar and copies the norm back
  after the timed update. Correctness and trainer losses/norms passed with
  `grad_scale_backend = device AdamW scalar`, but x3 TinyStories averaged
  `2627.288 ms`; keep the host scalar AdamW path.
- Precomputed device grad-scale hypothesis: an opt-in
  `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` branch computes `grad_scale` once in
  a one-thread device kernel, then AdamW reads that scalar. This improves over
  the per-thread device-sqrt variant but still averaged `2614.956 ms` over x3
  TinyStories steps, so the host scalar AdamW path remains the best current
  trainer route.
- Native-zero plus LayerNorm composition check:
  `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`,
  `LLMK_SM120_USE_CUDA_KERNEL_DRESIDUAL_ZERO`,
  `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and
  `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` passed focused correctness and
  trainer loss checks, but averaged `2617.466 ms` over x3 TinyStories steps.
  This rules out a positive interaction between the native CUDA zeroing
  near-miss and the LayerNorm backward one-block-per-SM setting under the
  current trainer state.
- AdamW helper guard source-size hypothesis: temporarily guarding unused
  opt-in AdamW helper kernels out of default builds kept the trainer on CUDA
  runtime zeroing and host scalar grad scale, but averaged `2617.405 ms` over
  x3 TinyStories steps. The patch was reverted, `test_adamw` passed after the
  default rebuild, and `train_gpt2cu` is restored to the default
  `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea` hash.
- LibTorch dresidual-zero plus scheduling check:
  `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1` with
  `CUDA_DEVICE_MAX_CONNECTIONS=1` passed all focused correctness smokes and
  trained normally with `dresidual_zero_backend = Torch C++`, but averaged
  `2500.425 ms` over x3 TinyStories steps. This is slower than both the
  stable `2493.133 ms` x10 baseline and the default `CUDA_DEVICE_MAX_CONNECTIONS=1`
  x3 recovery run, so the Torch C++ dresidual-zero trainer route remains
  rejected for promotion. Default binaries were rebuilt afterward and
  `train_gpt2cu` is restored to
  `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- LibTorch dresidual-zero plus fastest dInput/dGeLU composition:
  `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1` with direct cuBLAS dInput for
  attention projection and MLP-up plus TK fused dGELU dInput passed focused
  correctness, but averaged `2510.396 ms` over x3 TinyStories steps. This is
  slower than both component evidence and the stable x10 baseline, so the
  native fastest dInput/dGeLU stack should not be combined with the Torch C++
  residual clear. Default binaries were rebuilt afterward.
- MLP-projection dInput selector check: a narrow opt-in
  `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` branch now exists for the
  context-sensitive `C=3072, OC=768` row. It passed all focused correctness
  smokes and trainer loss checks, but averaged `2651.203 ms` over x3
  TinyStories steps. Keep the current cuBLASLt route for that row; the direct
  cuBLAS flip is not a training-speed improvement.
- Default native benchmark-warmup recheck:
  `codex_sm120_default_native_benchmark_warmup_x3_20260521` rebuilt the
  default/no-candidate trainer, ran all focused correctness smokes plus 95
  native benchmark rows before training, and averaged `2499.009 ms` over x3
  TinyStories steps with normal losses. This did not reproduce the older
  `codex_sm120_round_memory_shape_coverage_20260520` x3 average of
  `2490.206 ms` and remains `5.876 ms` slower than the stable x10 baseline, so
  it is drift evidence rather than a promotion.
- LibTorch grad-zero plus all-native winner composition:
  `SM120_USE_LIBTORCH_GRAD_ZERO=1` combined with direct cuBLAS dInput for
  attention projection and MLP-up, TK fused dGELU dInput, one-block-per-SM
  LayerNorm backward, and 1024-thread wide bias-add passed all focused smokes
  and averaged `2497.004 ms` over x3 TinyStories steps. This is the best
  composed x3 result in the current combination pass, beating the all-native
  composition by `1.174 ms`, but it is still `3.871 ms` slower than the stable
  x10 baseline. Keep it rejected for promotion; the default/no-candidate binary
  was rebuilt afterward.
- LibTorch grad-zero plus fastest dInput/dGeLU composition:
  `SM120_USE_LIBTORCH_GRAD_ZERO=1` with direct cuBLAS dInput for attention
  projection and MLP-up plus TK fused dGELU dInput, but without LayerNorm bwd1
  or wide-bias, passed all focused smokes and averaged `2496.595 ms` over x3
  TinyStories steps. This is the best composed x3 result so far, improving
  `0.409 ms` over the broader LibTorch/all-native composition, but it still
  trails the stable x10 baseline by `3.462 ms`, so it remains rejected for
  promotion and the default/no-candidate binary was rebuilt afterward.
- LibTorch grad-zero plus fastest dInput/dGELU with maxconn1:
  adding `CUDA_DEVICE_MAX_CONNECTIONS=1` to that composed stack passed all
  focused smokes but averaged `2497.422 ms` over x3 TinyStories steps. The
  runtime scheduling knob worsened the best dInput/dGELU composition by
  `0.828 ms`, so that larger mixed stack stays rejected.
- LibTorch grad-zero plus maxconn1 stability gate:
  the narrower stack, `SM120_USE_LIBTORCH_GRAD_ZERO=1` with native dInput/dGELU
  defaults and `CUDA_DEVICE_MAX_CONNECTIONS=1`, averaged `2492.843 ms` over x3
  TinyStories steps and briefly beat the stable x10 baseline by `0.290 ms`.
  The required x10 gate averaged `2497.047 ms`, which is `3.914 ms` slower
  than the stable baseline, so the short-run win is rejected and the default
  no-candidate binary was restored to `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- Attention dprep=3 current-stack retest:
  `LLMK_SM120_DPREP_WARPS=3` was retested against the current
  cuBLASLt-backed trainer mix because the older dprep=3 signal came from a
  different stack. The candidate passed all focused smokes and trained with
  normal losses, but averaged `2494.257 ms` over x3 TinyStories steps, which is
  `1.124 ms` slower than the stable `2493.133 ms` x10 baseline. Keep the
  default dprep setting; the default no-candidate binary was restored to
  `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- LibTorch grad-zero plus TK dGELU isolation:
  `SM120_USE_LIBTORCH_GRAD_ZERO=1` was combined only with
  `LLMK_SM120_USE_TK_FUSED_DGELU_DINP`, leaving out the direct-cuBLAS dInput
  selectors that already failed x10. The run passed all focused smokes and used
  `grad_zero_backend = Torch C++`, but averaged `2658.973 ms` over x3
  TinyStories steps. This rules out a hidden positive interaction between the
  two strongest x3-only component signals; default binaries were restored to
  `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- cuBLASLt workspace/results-width trainer check:
  `LLMK_SM120_CUBLASLT_WORKSPACE_MB=256` plus
  `LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS=16` passed all focused correctness
  smokes with the default CUDA runtime zeroing path, but averaged
  `2509.786 ms` over x3 TinyStories steps. Keep the default 128 MB workspace
  and 8-result lowest-waves cuBLASLt selection; the wider search does not
  improve trainer speed and the default binary was restored to
  `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- cuBLAS dInput plus attention dprep=3 combination:
  the closest x10 cuBLAS dInput selector near-miss was combined with the
  older attention `LLMK_SM120_DPREP_WARPS=3` near-miss using
  `LLMK_SM120_USE_CUBLAS_DINP_ATTPROJ`,
  `LLMK_SM120_USE_CUBLAS_DINP_FC`, and `LLMK_SM120_DPREP_WARPS=3`.
  The run passed all focused smokes and trained with normal losses, but
  averaged `2543.476 ms` over x3 TinyStories steps. The pair is `50.343 ms`
  slower than the stable x10 baseline, so no x10 gate is justified; default
  binaries were restored to
  `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- LibTorch grad-zero plus maxconn1 plus attention dprep=3 composition:
  `SM120_USE_LIBTORCH_GRAD_ZERO=1`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and
  `LLMK_SM120_DPREP_WARPS=3` passed all focused smokes and produced the best
  composed x3 result in this pass at `2486.639 ms`, beating the stable x10
  baseline by `6.494 ms`. The required x10 gate rejected it, averaging
  `2509.052 ms` with late steps at `2545.70` and `2562.27 ms`, so the short
  run was not stable. Keep the default CUDA runtime zeroing and default dprep
  settings; default binaries were restored to
  `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- LibTorch grad-zero plus attention dprep=3 without maxconn1:
  removing `CUDA_DEVICE_MAX_CONNECTIONS=1` from the previous best short-run
  composition improved the x3 smoke to `2485.322 ms`, beating stable x10 by
  `7.811 ms` with all focused smokes and normal losses. The x10 gate rejected
  it at `2497.490 ms`, which is `4.357 ms` slower than stable x10. This
  confirms maxconn1 amplified the late-step drift, but Torch C++ grad-zero plus
  dprep=3 still is not a stable promotion. Default binaries were restored to
  `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- LibTorch dresidual-zero plus attention dprep=3:
  `SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1` with `LLMK_SM120_DPREP_WARPS=3` passed
  all focused smokes and averaged `2488.597 ms` over x3 TinyStories steps,
  enough to require an x10 gate. The x10 gate averaged `2494.706 ms`, which is
  the closest recent composed x10 result but still `1.573 ms` slower than the
  stable x10 baseline. Keep the default CUDA runtime dresidual-zero path and
  default dprep setting; default binaries were restored to
  `cbcf72b7010de7d19f1b5a69c527d2e4d47aa5c26bc8d145db0f330d739675ea`.
- CUDA grad-zero plus LibTorch dresidual-zero plus attention dprep=3:
  replacing only the default gradient clear path in the prior closest
  Torch+dprep composition with `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO` and
  `LLMK_SM120_MEMORY_BLOCK_SIZE=1024` passed all focused smokes and averaged
  `2486.084 ms` over x3 TinyStories steps. The required x10 gate averaged
  `2494.320 ms`, improving the previous Torch+dprep x10 by `0.387 ms` but
  still landing `1.187 ms` slower than the stable `2493.133 ms` x10 baseline.
  Do not promote; default no-candidate binaries were restored to
  `66c0932a876c57052cdc07d5411e70c8235c237ae600ff7081b289aaa066f002`.
- LibTorch dresidual-zero plus attention dprep=3 plus LayerNorm bwd1:
  adding `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` to the closest
  Torch+dprep composition passed all focused smokes and produced the best x3
  result in the current composition pass at `2485.470 ms`, beating stable x10
  by `7.663 ms`. The required x10 gate averaged `2495.009 ms`, which is
  `1.876 ms` slower than the stable baseline, so the short-run win is rejected.
  Default no-candidate binaries were restored to
  `17970026dbad81d082f4249c4e927eba315032b3a698fdeb05125e8d73877f89`.
- Four-way CUDA/Torch/dprep/LayerNorm composition:
  combining CUDA-kernel grad zero, Torch C++ dresidual-zero, attention
  `LLMK_SM120_DPREP_WARPS=3`, `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`, and
  `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1` passed all focused smokes and
  produced the best x3 result in this pass at `2484.400 ms`, beating stable x10
  by `8.733 ms`. The x10 gate averaged `2493.735 ms`, the closest composed x10
  result so far but still `0.602 ms` slower than the stable baseline. Do not
  promote a non-win; default no-candidate binaries were restored to
  `d40f2eceabad9ec89cdf873479cf997a5bd780b34900e569eabcf9d072f96c34`.
- Four-way composition with source-default dprep=4:
  the same CUDA-kernel grad-zero, Torch C++ dresidual-zero, LayerNorm bwd1, and
  1024-thread memory block stack was retested with `LLMK_SM120_DPREP_WARPS=4`
  to check whether the source-default attention prep setting reduced late-step
  drift. It averaged `2486.240 ms` over x3, then `2493.970 ms` over x10. This
  is worse than the dprep=3 four-way gate and remains `0.837 ms` slower than
  the stable baseline, so keep the source default trainer unchanged; default
  no-candidate binaries were restored to
  `aa904a1a5d3ca24a9a370d410a3b73665e64a9a09e9344daaef633427997ad9e`.
- Four-way composition with 512-thread memory zero blocks:
  the closest dprep=3 four-way stack was retested with
  `LLMK_SM120_MEMORY_BLOCK_SIZE=512` to check whether the zero-kernel block
  size was responsible for the remaining x10 tail drift. It averaged
  `2485.324 ms` over x3, then `2494.177 ms` over x10. This is slower than the
  1024-thread four-way x10 and remains `1.044 ms` slower than the stable
  baseline, so no Torch/CUDA composed route is promoted; default no-candidate
  binaries were restored to
  `b350a6626f320b170be8d090d47716c169009b92f7b4b418c92f5a3ef0b99d2c`.
- Four-way composition with 256-thread memory zero blocks:
  the same dprep=3 four-way stack was retested with
  `LLMK_SM120_MEMORY_BLOCK_SIZE=256` to finish the zero-kernel block-size
  sweep. It averaged `2485.912 ms` over x3, then `2499.345 ms` over x10 with a
  step-6 tail spike. This is slower than both the 512-thread and 1024-thread
  gates and remains `6.212 ms` slower than the stable baseline, so the 1024
  block remains the best memory-block variant;
  default no-candidate binaries were restored to
  `4380d98cdadeb9dad2bbe85923ed4dc438499dc8d06ec3f68b9ce4fa4a3a0680`.
- Four-way composition with maxconn1:
  adding `CUDA_DEVICE_MAX_CONNECTIONS=1` to the best block1024 four-way stack
  produced the first repeatable x10 win. The x3 gate averaged `2481.166 ms`;
  the first x10 gate averaged `2490.940 ms`; the confirmation x10 averaged
  `2490.432 ms`. After promotion into the SM120 default build and
  `train-sm120.sh`, direct script timing averaged `2489.062 ms`, beating the
  stable baseline by `4.071 ms`.
- Promoted stack plus direct-cuBLAS dInput and TK dGELU:
  revisiting the prior fastest-row dInput/dGELU composition on top of the
  promoted stack produced an attractive x3 average of `2491.288 ms`, but the
  x10 gate averaged `2508.456 ms`. That is `19.394 ms` slower than promoted
  direct `train-sm120.sh` evidence and `15.323 ms` slower than stable x10, so
  the combination is rejected. Promoted-default binaries were rebuilt afterward
  to `8254ee2d1065c6c921b2492f7349d6c3144982f2d7700b18e819a789cbd7c49e`,
  and all nine focused CUDA smokes passed.
- Promoted stack plus fcproj direct-cuBLAS dInput:
  the fcproj-only direct-cuBLAS selector was escalated from the near-miss x3
  into a full x10 gate. The first x10 averaged `2490.019 ms`, beating the
  same-session promoted default by `43.465 ms` and stable x10 by `3.114 ms`,
  but still landing `0.957 ms` behind the direct promoted `train-sm120.sh`
  proof. The confirmation x10 averaged `2548.056 ms` with late-step spikes,
  so the selector is rejected as unstable and the promoted default remains the
  CUDA-kernel grad-zero, Torch C++ dresidual-zero, dprep3, block1024,
  LayerNorm-bwd1, maxconn1 stack. Promoted-default binaries were restored to
  `1edf1131bcc110f3b2cc16a4a090273f4920807f661547e864877f901c236c85`, and all
  nine focused CUDA smokes passed after restoration.
- Promoted stack plus async grad-norm scalar copy:
  `LLMK_SM120_ASYNC_GRAD_NORM_COPY` was retested on top of the final promoted
  CUDA/Torch/dprep/LayerNorm/maxconn stack. It passed all focused smokes and
  kept the expected promoted backends active, but x3 TinyStories averaged
  `2508.091 ms`. That is `19.029 ms` slower than the direct promoted
  `train-sm120.sh` proof and `14.958 ms` slower than stable x10, so the async
  readback route remains rejected and does not justify an x10 gate. Promoted
  default binaries were restored to
  `30e73fb995fbca0d18cdb0ec976da12908ac2cd34939cf3c81e9aafb3d1b31aa`, and all
  nine focused CUDA smokes passed after restoration.
- Promoted stack plus device AdamW grad-scale:
  `LLMK_SM120_DEVICE_GRAD_SCALE_ADAMW` was retested on top of the final
  promoted stack to close out the direct device-scalar AdamW route. Startup
  confirmed `grad_scale_backend = device AdamW scalar`, correctness passed, and
  losses stayed normal, but x3 TinyStories averaged `2528.446 ms`. That is
  `39.384 ms` slower than the direct promoted `train-sm120.sh` proof and
  `35.313 ms` slower than stable x10. The host-scalar AdamW route remains
  selected; the precomputed device scalar route is still the less-bad rejected
  device-grad-scale variant. Promoted default binaries were restored to
  `68e1745626dc3cdf2e6ccc825b8dd5fed70bb6ff7601c1f1730aacfcf07f0fc9`, and all
  nine focused CUDA smokes passed after restoration.
- Promoted memory store-policy sweep:
  the native memory wrapper's remaining store policies were tested because the
  promoted CUDA-kernel grad-zero route uses this path. `LLMK_SM120_MEMORY_STORE_POLICY=2`
  averaged `2592.754 ms` over x3, and `LLMK_SM120_MEMORY_STORE_POLICY=0`
  averaged `2593.304 ms` over x3. Both passed correctness but regressed by
  about `104 ms` versus promoted direct timing, so the default streaming-store
  policy `1` remains selected.
- Promoted stack plus disabled direct-cuBLAS backward selector at maxconn8:
  composing `LLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM` with
  `CUDA_DEVICE_MAX_CONNECTIONS=8` averaged `2534.380 ms` over x3. That recovered
  `7.253 ms` versus today's exact direct-script regression control, but it was
  `13.163 ms` slower than the same candidate at maxconn1 and remained
  `45.317 ms` slower than promoted direct `train-sm120.sh` evidence. The
  candidate is rejected, and promoted-default binaries were restored to
  `0d638c6cd459e201ef06a0f8854c3d51e96c5769a67570c995c8cd209a89cf66` with all
  nine focused CUDA smokes passing after restoration.
- Promoted stack plus disabled direct-cuBLAS backward selector and attention
  bwd32:
  composing `LLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM` with
  `LLMK_SM120_ATTN_BWD_BLOCK=32` averaged `2553.930 ms` over x3. The interaction
  is negative: it is `32.714 ms` slower than the disabled-backward-selector
  candidate alone and `51.429 ms` slower than the attention-bwd32 candidate
  alone. Focused attention backward also worsened to `2872.892 us`, so this
  route is rejected without x10 gating. Promoted-default binaries were restored
  to `267fdc471f8a924751e551ed09bc98811092e0b3a7865d05a866ef1f4443f35a` with
  all nine focused CUDA smokes passing after restoration.
- Promoted stack plus disabled direct-cuBLAS backward selector and TK dGELU:
  composing `LLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM` with
  `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` averaged `2536.936 ms` over x3. It is
  `4.696 ms` faster than today's exact direct-script regression control, but
  `15.720 ms` slower than the disabled-backward-selector candidate alone and
  `34.178 ms` slower than the promoted TK dGELU candidate alone. The focused
  `dInp+dGeLU` row still favors TK (`1796.610 us` versus cuBLASLt fused
  `1817.670 us`), but the trainer composition does not. Promoted-default
  binaries were restored to
  `1d89cc669368b2d163f3f0edf994ead1aeab075b833b056d4f99de4b18a33d37` with all
  nine focused CUDA smokes passing after restoration.
- Promoted stack with cuBLASLt GEMM disabled:
  the wholesale no-cuBLASLt trainer build was correctness-rejected before
  training. `test_matmul` passed 9/10 checks but failed the GPT-2 fcproj fused
  dGELU backward route at the tolerance boundary, while the restored promoted
  cuBLASLt stack passed the same route. The active promoted-default
  `train_gpt2cu` hash after restoration is
  `0817d8477161c7f1b829e1b42563f7038540b39844802f88f6570c7c6488b1fa`, with all
  nine focused CUDA smokes passing. Keep the cuBLASLt-backed promoted trainer
  route selected.
- Recovered-band LibTorch grad-zero retest:
  the trainer-callable Torch C++ grad-zero route was retested on top of the
  promoted stack because the current project-wide matrix shows a narrow
  runtime-row Torch win. It passed correctness and activated
  `grad_zero_backend = Torch C++`, but x3 training averaged `2532.897 ms`,
  which is `53.503 ms` slower than the recovered selected-stack x3 and
  `63.054 ms` slower than the `new-goal.md` target. Promoted-default binaries
  were restored to
  `4a383d26106f39decf9952eb1283b2bf7e22f5088bcbcdf63cc36cfee7b35c81`, with all
  nine focused CUDA smokes passing. Keep CUDA-kernel grad-zero selected; Torch
  remains promoted only for dresidual zero.
- Exact `train-sm120.sh` after restore:
  the restored promoted stack ran with CUDA-kernel grad-zero, Torch C++
  dresidual-zero, host scalar AdamW grad scale, and `CUDA_DEVICE_MAX_CONNECTIONS=1`.
  The x10 average was `2496.657 ms`, which is back out of the severe slow band
  but still `7.595 ms` slower than the promoted direct proof and `26.814 ms`
  slower than the `new-goal.md` first-three target. This points to runtime
  drift/noise at the selected stack rather than a hidden switch to a rejected
  Torch or no-cuBLASLt component.
- CUDA runtime grad-zero x10 gate:
  the recovered-band CUDA runtime grad-zero near-miss was escalated to x10
  before considering any replacement of the selected CUDA-kernel grad-zero
  route. It passed all nine focused smokes and startup confirmed
  `grad_zero_backend = CUDA runtime`, but x10 averaged `2489.146 ms`, which is
  `0.084 ms` slower than promoted direct proof, while visible first3 averaged
  `2481.347 ms`, which is `11.504 ms` slower than `new-goal.md`. The selected
  CUDA-kernel grad-zero route remains active; the restored binary is
  `0452da6344d3144b24d4c213ddcacf34dbc935f90e04a5288193dc8c42f7f15e`, with all
  nine focused smokes passing after restoration.
- Direct follow-up rerun after user fast-band paste:
  the exact `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh` path was rerun on
  an idle GPU with active binary
  `0452da6344d3144b24d4c213ddcacf34dbc935f90e04a5288193dc8c42f7f15e`. Startup
  confirmed the selected CUDA-kernel grad-zero, Torch C++ dresidual-zero, host
  scalar stack, and x10 averaged `2489.367 ms` with visible first3
  `2482.793 ms`. This again missed the user's pasted `2461.450 ms` first-three
  band, so it is recorded as a reproduction miss rather than promotion
  evidence.
- Refreshed attention integration lead:
  reran native packed TK, cuDNN, Torch packed/materialized, and Torch qkv-layout
  attention probes in the target conda context. Native packed TK measured
  `785.153/2734.630 us` forward/backward (`3519.783 us` total). Separated
  cuDNN (`3063.465 us`) and Torch (`2753.362 us`) remain faster reference rows,
  but the trainer-compatible routes do not beat TK: cuDNN packed totals
  `3613.122 us`, Torch packed totals `5226.573 us`, Torch materialized packed
  totals `5458.822 us`, and the qkv-layout routes are `6960.832-8421.472 us`.
  No cuDNN or Torch attention trainer route is justified from this refresh.
- Exact `new-goal.md` command control:
  ran the literal three-step command from `new-goal.md` against active selected
  binary `0452da6344d3144b24d4c213ddcacf34dbc935f90e04a5288193dc8c42f7f15e`.
  Startup confirmed the selected CUDA-kernel grad-zero, Torch C++
  dresidual-zero, host scalar stack. Step timings were `2486.14`, `2479.69`,
  and `2483.73 ms`; trainer average was `2481.710 ms`, and visible first3 was
  `2483.187 ms`. That is still `13.344 ms` slower than `new-goal.md`, so the
  `-x 3` command shape does not explain the gap. Generated step-3 checkpoints
  were removed.
- Promoted-stack profiler-disable retest:
  closed the remaining cheap source-runtime knob on the final selected stack by
  adding `LLMK_DISABLE_CUDA_PROFILER` to the CUDA-kernel grad-zero, Torch C++
  dresidual-zero, dprep3, memory block1024, LayerNorm-bwd1, maxconn1 build. It
  passed all nine focused CUDA smokes, but trained at `2482.823 ms` average
  with visible first3 `2484.143 ms`, slightly slower than the exact selected
  x3 control and still `14.300 ms` behind `new-goal.md`. The candidate binary
  was `105e95d7...`; the selected binary was restored to `407bd0a4...`, with
  all nine focused smokes passing after restoration.
- Idle-GPU selected-stack rerun after user fast-band paste:
  reran the exact `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh` path with
  active binary `407bd0a4...`. Pre-run `nvidia-smi` showed no running processes,
  P8, 47 W, 2641 MiB resident memory, and 1% GPU utilization. Startup confirmed
  CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar grad scale, and
  ZeRO stage 1. Step timings were `2486.88`, `2479.96`, `2486.13`, `2486.74`,
  `2491.03`, `2492.87`, `2494.46`, `2499.02`, `2500.74`, and `2501.64 ms`;
  x10 averaged `2492.511 ms`, first3 averaged `2484.323 ms`, and first5
  averaged `2486.148 ms`. This still misses both `new-goal.md` first3 by
  `14.480 ms` and the user's pasted first3 by `22.873 ms`, so it is recorded as
  another reproduction miss, not promotion evidence.
- Matmul backward-bias reduction block-size sweep:
  tested the previously unrecorded `LLMK_SM120_BIAS_BLOCK_SIZE` hook on top of
  the selected promoted stack. This is the dbias reduction inside
  `matmul_backward`, not the already-rejected bias-add block-size path. Both
  candidates passed all nine focused CUDA smokes, but neither improved training:
  `1024` averaged `2486.839 ms` with first3 `2489.267 ms`, and `256` averaged
  `2492.912 ms` with first3 `2495.430 ms`. Both miss the `new-goal.md` first3
  target, so the default 512-thread dbias reduction block remains selected.
  The selected default was rebuilt afterward to `train_gpt2cu` hash
  `a60e97a6...`, with all nine focused smokes passing.
- Direct selected-stack rerun after user 2.46s-band report:
  reran the exact `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh` path with
  active binary `a60e97a6...`. Pre-run `nvidia-smi` showed no running compute
  processes, P8, 47.27 W, 2660 MiB resident memory, and 1% GPU utilization.
  Startup confirmed CUDA-kernel grad-zero, Torch C++ dresidual-zero, host
  scalar grad scale, and ZeRO stage 1. Step timings were `2486.77`, `2479.92`,
  `2485.18`, `2488.32`, `2492.04`, `2492.45`, `2498.59`, `2502.10`,
  `2501.67`, and `2502.72 ms`; x10 averaged `2493.667 ms`, first3 averaged
  `2483.957 ms`, and first5 averaged `2486.446 ms`. This misses both
  `new-goal.md` first3 by `14.114 ms` and the user's pasted first3 by
  `22.507 ms`, so it is recorded as another reproduction miss, not promotion
  evidence.
- fcproj direct-cuBLAS plus runtime grad-zero composition:
  tested the unrecorded interaction between `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ`
  and CUDA runtime grad-zero on top of the selected cuBLASLt/Torch-dresidual
  stack. The candidate passed all nine focused CUDA smokes and validated with
  native benchmarks, but TinyStories x3 averaged `2490.053 ms` with first3
  `2491.610 ms`. This is `21.767 ms` slower than `new-goal.md` first3 and
  `7.653 ms` slower than the latest direct selected-stack rerun, so it is
  rejected without x10 gating. The selected default was restored to
  `train_gpt2cu` hash `5f5decae...`, with all nine focused smokes passing.
- fcproj direct-cuBLAS plus precomputed AdamW grad scale:
  tested `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` with
  `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW`. The candidate passed all nine
  focused CUDA smokes and validated, but TinyStories x3 averaged
  `2485.572 ms` with first3 `2486.267 ms`. This improves over the
  fcproj+runtime-grad-zero interaction, but it is still `16.424 ms` slower
  than `new-goal.md`, `2.310 ms` slower than the latest direct selected-stack
  rerun, and `6.877 ms` slower than the recovered selected-stack x3 control.
  The selected default was restored to `train_gpt2cu` hash `067f00c3...`, with
  all nine focused smokes passing.
- TK fused dGELU plus CUDA runtime grad-zero:
  tested `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` with CUDA runtime grad-zero after
  the native fcproj dInput+dGeLU row showed a small TK win. Correctness and
  native benchmarks passed, but TinyStories x3 averaged `2496.712 ms` with
  first3 `2496.960 ms`, which is `27.117 ms` slower than `new-goal.md` and
  worse than the latest direct selected-stack rerun. The selected default was
  restored to `train_gpt2cu` hash `47f15e24...`, with all nine focused smokes
  passing.
- Direct selected-stack rerun after user 2.46s-band follow-up:
  reran `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh` against active binary
  `47f15e24...` after a fresh `nvidia-smi` showed no running compute processes.
  Startup confirmed CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar
  grad scale, and ZeRO stage 1. Step timings were `2488.58`, `2483.47`,
  `2485.15`, `2488.12`, `2490.31`, `2492.06`, `2500.91`, `2509.12`,
  `2503.65`, and `2505.38 ms`; trainer x10 averaged `2495.353 ms`, visible
  x10 averaged `2494.675 ms`, first3 averaged `2485.733 ms`, and first5
  averaged `2487.126 ms`. This still misses both `new-goal.md` first3 by
  `15.890 ms` and the user's pasted first3 by `24.283 ms`, so it remains a
  reproduction miss.
- Disabled direct-cuBLAS backward plus CUDA runtime grad-zero:
  composed `LLMK_SM120_DISABLE_CUBLAS_BACKWARD_GEMM` with the CUDA runtime
  grad-zero route because each had a narrow recovered/degraded-band signal.
  The candidate passed correctness and native benchmarks, but TinyStories x3
  averaged `2518.967 ms` with visible first3 `2520.663 ms`, which is
  `50.820 ms` slower than `new-goal.md` and `34.930 ms` slower than the latest
  direct selected-stack rerun. The interaction is rejected; the selected default
  was restored to `train_gpt2cu` hash `1cfb77e2...`, with all nine focused
  smokes passing.
- fcproj direct-cuBLAS plus no-ZeRO:
  composed `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` with `TRAIN_ZERO_STAGE=0` to test
  whether two near-current rows interact positively. The candidate passed
  correctness and native benchmarks, but TinyStories x3 averaged `2491.644 ms`
  with visible first3 `2493.310 ms`. That is `23.467 ms` slower than
  `new-goal.md`, `7.577 ms` slower than the latest direct selected-stack rerun,
  and slower than the better fcproj/precompute and no-ZeRO-alone rows. The
  selected default was restored to `train_gpt2cu` hash `e43a87cf...`, with all
  nine focused smokes passing.
- TK fused dGELU plus precomputed AdamW grad scale:
  composed `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` with
  `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW` because the TK row still wins the
  focused fcproj dInput+dGeLU benchmark and precompute was a near-current
  trainer route. The candidate passed correctness and native benchmarks, but
  TinyStories x3 averaged `2498.933 ms` with visible first3 `2498.490 ms`.
  This is `28.647 ms` slower than `new-goal.md` and worse than both component
  rows, so it is rejected. The selected default was restored to `train_gpt2cu`
  hash `e42983fb...`, with all nine focused smokes passing.
- fcproj direct-cuBLAS plus precomputed AdamW grad scale plus no-ZeRO:
  composed `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ`,
  `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW`, and `TRAIN_ZERO_STAGE=0` because
  the component rows were close enough to require a real trainer interaction
  test. The candidate passed correctness and native benchmarks, but TinyStories
  x3 averaged `2490.259 ms` with visible first3 `2491.700 ms`. This improves
  over fcproj+no-ZeRO alone, but remains `21.857 ms` slower than `new-goal.md`
  and slower than the better component rows, so it is rejected. The selected
  default was restored to `train_gpt2cu` hash `a60e97a6...`, with all nine
  focused smokes passing.
- Direct selected-stack rerun after user follow-up:
  reran `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh` against active binary
  `a60e97a6...` after a fresh `nvidia-smi` showed no running compute processes.
  Startup confirmed CUDA-kernel grad-zero, Torch C++ dresidual-zero, host scalar
  grad scale, and ZeRO stage 1. Step timings were `2489.27`, `2483.65`,
  `2487.86`, `2488.85`, `2493.01`, `2496.94`, `2500.06`, `2501.60`,
  `2504.85`, and `2505.55 ms`; trainer x10 averaged `2495.819 ms`, visible
  x10 averaged `2495.164 ms`, first3 averaged `2486.927 ms`, and first5
  averaged `2488.528 ms`. This does not reproduce the user's faster
  `2456-2465 ms` band, so it remains a reproduction miss rather than promotion
  evidence.
- CUDA-runtime grad-zero plus precomputed AdamW grad scale:
  composed the recovered-band CUDA-runtime gradient-zero route with
  `LLMK_SM120_PRECOMPUTE_GRAD_SCALE_ADAMW`, while keeping the promoted cuBLASLt,
  attention, dprep3, LayerNorm-bwd1, memory-block1024, Torch C++ dresidual, and
  ZeRO-1 stack. Correctness and native benchmarks passed, but TinyStories x3
  averaged `2493.946 ms` with visible first3 `2495.577 ms`. This is slower than
  both component rows and the selected-stack controls, so it is rejected. The
  selected default was restored to `train_gpt2cu` hash `a60e97a6...`, with all
  nine focused smokes passing.
- CUDA-runtime grad-zero plus no-ZeRO:
  composed CUDA-runtime gradient-zero with `TRAIN_ZERO_STAGE=0` because both
  individual rows were near-current. Correctness and native benchmarks passed,
  but TinyStories x3 averaged `2494.380 ms` with visible first3
  `2495.407 ms`. This is slower than `new-goal.md`, the latest direct
  selected-stack rerun, and the better component rows, so it is rejected. The
  selected default was restored to `train_gpt2cu` hash `a60e97a6...`.
- Direct selected-stack rerun after user follow-up:
  reran `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh` against active binary
  `a60e97a6...`. Startup confirmed CUDA-kernel grad-zero, Torch C++ dresidual
  zero, host scalar grad scale, and ZeRO stage 1. Step timings were `2491.29`,
  `2486.02`, `2487.97`, `2491.41`, `2494.46`, `2499.49`, `2501.68`,
  `2503.84`, `2506.13`, and `2507.12 ms`; trainer x10 averaged
  `2497.571 ms`, visible x10 averaged `2496.941 ms`, first3 averaged
  `2488.427 ms`, and first5 averaged `2490.230 ms`. This still does not
  reproduce the user's faster `2456-2465 ms` band, so it remains a reproduction
  miss rather than promotion evidence.
- Direct selected-stack rerun after user noted GPU contention:
  reran `bash train-sm120.sh` against active binary `a60e97a6...`. Startup
  confirmed CUDA-kernel grad-zero, Torch C++ dresidual zero, host scalar grad
  scale, fused GELU, and ZeRO stage 1. Step timings were `2489.65`, `2485.15`,
  `2487.15`, `2495.33`, `2499.15`, `2498.13`, `2502.87`, `2503.86`,
  `2505.36`, and `2507.66 ms`; trainer x10 averaged `2498.294 ms`, visible
  x10 averaged `2497.431 ms`, first3 averaged `2487.317 ms`, and first5
  averaged `2491.286 ms`. This remains slower than the user's fresh first5
  observation and `new-goal.md`; treat it as another reproduction miss, not
  promotion evidence.
- Direct selected-stack rerun after the user's second retry request:
  reran `bash train-sm120.sh` against active binary `e2c85370...`. Startup
  confirmed CUDA-kernel grad-zero, Torch C++ dresidual zero, host scalar grad
  scale, fused GELU, and ZeRO stage 1. Step timings were `2488.09`, `2483.44`,
  `2487.56`, `2489.60`, `2491.58`, `2499.00`, `2500.83`, `2503.03`,
  `2503.78`, and `2505.35 ms`; trainer x10 averaged `2496.018 ms`, visible
  x10 averaged `2495.226 ms`, first3 averaged `2486.363 ms`, and first5
  averaged `2488.054 ms`. This is the best recent Codex rerun in the restored
  selected stack, but it still misses the user's `~2462 ms` first-five band and
  does not provide promotion evidence.
- TK approximate dGELU trainer route:
  tried `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` with
  `LLMK_SM120_APPROX_DGELU_TANH=1` because the fcproj dInput+dGeLU microbench
  was one of the remaining trainer-callable rows worth closing. The candidate
  built, but `test_matmul` failed the GPT-2 fcproj fused dGELU dInput row with
  `max abs diff = 0.5000` at tolerance `0.50`, so no benchmark or TinyStories
  training timing was accepted. The selected default was restored to
  `train_gpt2cu` hash `4374a593...`, with all nine focused smokes passing.
- Direct selected-stack control after TK approximate dGELU restore:
  reran `bash train-sm120.sh` against active binary `4374a593...`. Startup
  confirmed CUDA-kernel grad-zero, Torch C++ dresidual zero, host scalar grad
  scale, fused GELU, and ZeRO stage 1. Step timings were `2490.11`, `2484.31`,
  `2487.35`, `2491.20`, `2495.80`, `2501.34`, `2504.18`, `2507.46`,
  `2507.20`, and `2508.05 ms`; trainer x10 averaged `2498.545 ms`, visible
  x10 averaged `2497.700 ms`, first3 averaged `2487.257 ms`, and first5
  averaged `2489.754 ms`. This confirms the selected stack was restored, but
  remains a reproduction miss versus the user's `~2462 ms` band.
- CUDA-runtime grad-zero plus dprep=2:
  composed the close CUDA-runtime gradient-zero route with the dprep=2
  attention-prep setting, without adding already-regressed fcproj, TK dGELU,
  precomputed-grad-scale, or no-ZeRO changes. The candidate passed correctness
  and native benchmarks, but TinyStories x3 averaged `2497.146 ms` with visible
  first3 `2498.250 ms`. This is slower than `new-goal.md`, the latest direct
  selected-stack rerun, runtime-grad-zero x10 first3, and dprep=2 alone, so it
  is rejected. The selected default was restored to `train_gpt2cu` hash
  `a60e97a6...`, with all nine focused smokes passing.
- CUDA-runtime grad-zero plus profiler-disable:
  composed the closest non-selected CUDA-runtime gradient-zero route with
  `LLMK_DISABLE_CUDA_PROFILER`. Correctness and native benchmarks passed, but
  TinyStories x3 averaged `2496.375 ms` with visible first3 `2496.877 ms`.
  Removing profiler calls did not rescue runtime grad-zero: the candidate is
  slower than `new-goal.md`, the latest direct selected-stack rerun,
  runtime-grad-zero x10 first3, and profiler-disable alone. The selected
  default was restored to `train_gpt2cu` hash `a60e97a6...`, with all nine
  focused smokes passing.
- cuBLASLt 64 MiB workspace:
  tried `LLMK_SM120_CUBLASLT_WORKSPACE_MB=64` on top of the selected cuBLASLt,
  packed-QKV attention, CUDA-kernel grad-zero, Torch C++ dresidual-zero,
  dprep3, LayerNorm-bwd1, memory-block1024, maxconn1, and ZeRO-1 stack.
  Correctness, native benchmarks, and artifact validation passed, but
  TinyStories x3 averaged `2498.071 ms` with visible first3 `2498.130 ms`.
  This is slower than `new-goal.md`, the latest direct selected-stack rerun,
  and the user's pasted fast band, so it is rejected. The selected default was
  restored to `train_gpt2cu` hash `cef1ac4a...`, with all nine focused smokes
  passing.
- Direct selected-stack rerun after the user's latest retry request:
  reran `bash train-sm120.sh` against active binary `cef1ac4a...`. Startup
  confirmed CUDA-kernel grad-zero, Torch C++ dresidual zero, host scalar grad
  scale, fused GELU, and ZeRO stage 1. Step timings were `2492.33`, `2486.34`,
  `2489.37`, `2494.40`, `2498.75`, `2501.46`, `2502.95`, `2505.92`,
  `2507.38`, and `2509.17 ms`; trainer x10 averaged `2499.528 ms`, visible
  x10 averaged `2498.807 ms`, first3 averaged `2489.347 ms`, and first5
  averaged `2492.238 ms`. The selected backend mix is active, but this still
  misses the user's `~2462 ms` first-five band and remains reproduction-miss
  evidence rather than promotion evidence.
- Direct selected-stack reruns after the user's pasted no-contention sample:
  reran `./train-sm120.sh` twice against active binary `cef1ac4a...`. The first
  run produced trainer x10 `2500.735 ms`, visible x10 `2500.060 ms`, first3
  `2490.597 ms`, and first5 `2493.410 ms`; a post-run `nvidia-smi` snapshot
  showed no running GPU processes. The second immediate run produced trainer
  x10 `2509.576 ms`, visible x10 `2508.785 ms`, first3 `2501.197 ms`, and
  first5 `2503.640 ms`. The user's pasted first-five average is
  `2462.236 ms`, so both Codex reruns are reproduction misses in the same
  selected stack and do not change the promotion decision.
- Microbatch/accumulation sweep:
  tested whether the reported `estimated maximum batch size: 73` can improve
  selected-stack TinyStories throughput. `B=73/GA=7` fits but is memory
  saturated at `32397/32606 MiB` and drops to `184183 tok/s`. `B=70/GA=7`
  lowers visible step time to `2390.367 ms` only by processing fewer tokens;
  normalized throughput is `209909 tok/s`, effectively tied with the latest
  selected direct rerun. `B=65/GA=8` processes slightly more tokens than the
  script but drops to `209189 tok/s`. `B=71/GA=7` had a small x3 edge but the
  x10 gate fell to `209216 tok/s`; `B=72/GA=7` was worse than `B=71`.
  `B=68/GA=8` also showed a small x3 edge, then failed the x10 gate at
  `209380 tok/s`. Keep `train-sm120.sh` at `B=64`, `grad_accum_steps=8`,
  `total_batch_size=524288`.
- Current optional-stack refresh plus LibTorch grad-zero gate:
  refreshed native, Torch, Triton, cuDNN, CuTeDSL, and LibTorch benchmark
  coverage in `scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522`
  with `210` benchmark rows and full `43/43` Torch objective coverage. The only
  trainer-callable refreshed Torch-family row worth closing was the grad-buffer
  memset edge, so I ran `SM120_USE_LIBTORCH_GRAD_ZERO=1` with the selected fast
  stack. It passed correctness and had a small same-session x3 edge
  (`2500.630 ms` versus selected control `2504.322 ms`), then failed the x10
  gate at `2505.490 ms`. The selected stack was restored to CUDA-kernel
  grad-zero plus Torch C++ dresidual-zero; active `train_gpt2cu` hash is
  `ad9c7d14...`. Keep CUDA-kernel grad-zero selected.
- Current selected-stack audit refresh:
  ran `codex_sm120_selected_current_audit_x10_20260522` with correctness,
  native benchmarks, stack probe, artifact validation, and 10 TinyStories
  steps. The selected stack was active: CUDA-kernel grad-zero, Torch C++
  dresidual-zero, host scalar grad scale, packed-QKV TK attention, ZeRO stage
  1. The trainer averaged `2511.403 ms`, visible x10 `2510.694 ms`, first3
  `2501.750 ms`, and first5 `2505.916 ms`. The regenerated current audit passed
  `132` checks with `0` active promotion candidates, but this remains a
  slow-band reproduction miss versus the user's `2462.236 ms` first-five
  sample.
- LibTorch grad-zero plus all-native winners x10 gate:
  x10-gated the best previous ungated composed route in
  `codex_sm120_combo_libtorch_grad_zero_all_native_winners_x10_20260522`.
  Startup confirmed Torch C++ grad-zero and CUDA-runtime dresidual-zero, with
  direct-cuBLAS dInput selectors, TK fused dGELU, LayerNorm-bwd1, and wide
  bias-add enabled. Correctness and native benchmark validation passed, and the
  focused fcproj dInput+dGeLU row still favored TK (`1741.99 us` versus
  cuBLASLt-fused `1853.93 us`), but TinyStories x10 averaged `2519.610 ms`.
  Rejected: it is slower than the current selected x10 control by `8.207 ms`;
  selected binary restored to `d37c652b...`.
- fcproj direct-cuBLAS current rerun plus exact wrapper rerun:
  reran `LLMK_SM120_USE_CUBLAS_DINP_FCPROJ` in
  `codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522` after the user
  reported a faster no-contention sample. Correctness and native benchmark
  validation passed, but TinyStories x10 averaged `2528.994 ms`, slower than
  the current selected x10 control by `17.591 ms`, so it is rejected. I restored
  the selected CUDA-kernel grad-zero plus Torch C++ dresidual-zero stack and
  reran the exact `./train-sm120.sh` wrapper; startup confirmed the selected
  stack, but the run averaged `2572.516 ms` with step 5-7 spikes. The post-run
  GPU snapshot was idle (`P8`, `1%` GPU utilization), so this is a slow-band
  reproduction miss rather than evidence for changing kernels.
- Current optional-stack refresh with Torch/LibTorch:
  refreshed the full optional backend matrix in
  `codex_sm120_optional_refresh_current2_20260522`. The round passed all nine
  focused smokes and validated `210` benchmark rows, `43/43` Torch objective
  rows, LibTorch runtime parity, and the LibTorch trainer-link probe. The
  regenerated current audit passed `132` checks with `0` active promotion
  candidates. Torch still wins only for separated-Q/K/V attention
  (`556.565/2160.624 us` fwd/bwd), while trainer-layout TorchPacked is slower
  than packed TK (`5145.650 us` total versus `3481.306 us`), and cuDNNPacked is
  also slower (`3555.443 us`). LibTorch memory rows remain resolved by prior
  x10 trainer-route rejections. No TinyStories candidate was justified from
  this refresh; selected binary restored to `e1be7de3...`.
- Runtime grad-zero current x10 gate:
  reran the selected SM120 stack with CUDA runtime grad-zero replacing the
  CUDA-kernel grad-zero route while keeping Torch C++ dresidual-zero. The
  harness round `codex_sm120_runtime_grad_zero_current2_x10_20260522` passed
  all nine focused smokes, accepted `95` native benchmark rows, and averaged
  `2476.256 ms` over x10 with visible first3 `2467.690 ms`. The exact
  `CUDA_DEVICE_MAX_CONNECTIONS=1 ./train-sm120.sh` wrapper then averaged
  `2475.111 ms` x10, visible x10 `2474.671 ms`, first3 `2466.643 ms`, and
  first5 `2469.284 ms`. This is the best current Codex-run script result and
  beats the promoted direct proof by `13.951 ms`, but still misses the user's
  pasted `2462.236 ms` first-five sample. Promoted CUDA runtime grad-zero as
  the SM120 fast-trainer default; the default rebuild omitted
  `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO` and produced active binary
  `6ed5e22c...`. Keep Torch C++ dresidual-zero selected and keep
  Torch/LibTorch grad-zero rejected.
- Runtime grad-zero default audit x10:
  reran the promoted default as
  `codex_sm120_runtime_grad_zero_default_audit_x10_20260522` with stack-probe
  enabled. The build omitted `LLMK_SM120_USE_CUDA_KERNEL_GRAD_ZERO`, kept
  `LLMK_SM120_USE_LIBTORCH_DRESIDUAL_ZERO`, passed all nine focused smokes,
  accepted `95` native benchmark rows, recorded `9` backend stacks and `168`
  family-stack rows, and averaged `2473.342 ms` x10 with first3
  `2465.353 ms`. The exact post-rebuild wrapper proof on binary
  `6ed5e22c...` was faster: `2468.122 ms` x10, first3 `2461.187 ms`, first5
  `2463.022 ms`. Regenerated current selection/audit now points at this
  native round plus `codex_sm120_optional_refresh_current2_20260522` and passes
  `132` audit checks with `0` active promotions. Active binary after the full
  audit rebuild is `4bfe515d...`; the follow-up exact wrapper run on that
  binary averaged `2465.890 ms` x10, first3 `2458.770 ms`, and first5
  `2460.134 ms`, beating the user's pasted `2462.236 ms` first-five sample.
