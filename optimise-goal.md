# Goal Kickoff - Fastest Kernel Stack for GPT-2, Starting with SM120

This file expands the project goal from "port `llm.c` onto ThunderKittens" to
"select the fastest correct kernel implementation for every GPT-2 training
operator." The first target is SM120 / RTX 5090 because it is locally testable.
H100 remains relevant for the original ThunderKittens v1 parity work, but this
optimization track is allowed to mix kernel stacks when that wins.

Allowed stacks:

- ThunderKittens 2.0
- cuBLAS
- cuBLASLt
- cuDNN
- Triton
- Torch
- CuTeDSL
- Plain CUDA when it remains the fastest or safest option

The end state is a shape-aware mixed backend where every GPT-2 kernel family
uses the fastest correct implementation for that architecture and workload.
Do not force a pure-TK result if cuBLASLt, cuDNN, Triton, Torch, CuTeDSL, or
plain CUDA is faster.

## Current Baseline

The current checked-in project still has two overlapping goals:

- `goal.md` is the v1 port tracker. It documents the feature-for-feature
  ThunderKittens port and still has pending H100 runtime/parity gates.
- This file is the v2 optimization kickoff. It starts from the working SM120
  path and broadens the allowed backend set.

The current SM120 baseline is the restored RTX 5090 path documented in
`docs/sm120-rtx5090-baseline.md`:

- `DEVICE_ARCH=SM120`
- `SM120_USE_CUBLASLT_GEMM=1`
- cuBLASLt-backed GEMM fallback enabled by default
- SM120 packed-QKV attention enabled
- FP32 master weights disabled on SM120 by default (`-w 0`)
- fused GELU enabled for the cuBLASLt trainer build (`gelu_fusion 1`)
- 512-thread SM120 bias reduction default

Recorded evidence:

- llm.c baseline from `new-goal.md`: step times `3091.12`, `2679.79`,
  `2683.77 ms`, roughly `195k tok/s` after warmup.
- Restored documented SM120 baseline: about `2508 ms` over the 3-step
  TinyStories smoke, roughly `209k tok/s`.
- `new-goal.md` also records a better in-repo run: about `2469 ms` per step,
  roughly `212k tok/s`.
- Current `log124M/5090_S/main.log` shows a 10-step run with decreasing train
  loss from `11.0324` to `9.5886`.

Future promotions must rerun the smoke in the same environment before claiming
a new best. Historical notes are useful, but fresh timing wins.

## What Has Been Tried

The git log and changelog show extensive SM120 tuning. Treat these as prior art
and do not blindly repeat them without a new hypothesis.

Pure ThunderKittens / SM120 work:

- SM120 warp-scope GEMM routes for GPT-2 forward, dInput, dWeight, huge-N
  LM-head, fused bias+GELU, and fused dGELU.
- Shape-specific routes for qkv, attention projection, MLP up-projection,
  MLP projection, and LM-head rows.
- Tuning of `SUPER_M`, N96 routes, huge-N K tiles, dWeight split-K, direct
  B-column dInput, fused dInput+dGELU, in-place layout swaps, approximate dGELU
  tanh, cache/load policy, and ptxas options.
- SM120 packed-QKV attention forward/backward with block-size and prep-warp
  experiments.
- LayerNorm fallback and block-size experiments.

Accepted or useful outcomes:

- SM120 packed-QKV attention is a useful fast path.
- cuBLASLt is currently the best GEMM provider for most material GPT-2 rows.
- TK can be competitive on selected rows, but isolated microbench wins often did
  not improve the end-to-end TinyStories smoke.
- Plain CUDA remains appropriate for small elementwise/reduction kernels unless
  a fused backend proves faster end-to-end.
- `best_runs.md` is the right place to keep the live scoreboard.

Rejected patterns seen repeatedly:

- Candidate compiles that failed `test_matmul` or `test_attention`.
- Candidates that hit ptxas shared-memory limits.
- Candidates that triggered illegal memory access in microbenchmarks.
- Candidates that improved one GEMM row while regressing qkv, dWeight,
  fcproj, LM-head, or the final TinyStories smoke.
- cuBLASLt heuristic/cache variants that did not beat the restored baseline.
- Attention block/prep variants that passed smoke but slowed the trainer.

## What Is Left To Do

Build a comparable benchmark and selection matrix for every GPT-2 kernel family.
Every row needs correctness, timing, config, commit, and log evidence.

Kernel families to cover:

- GEMM forward: qkv, attention projection, MLP up, MLP projection, LM-head.
- GEMM backward dInput for the same shapes.
- GEMM backward dWeight for the same shapes, including accumulated paths.
- Bias add, bias gradient reduction, GELU, fused bias+GELU, and fused dGELU.
- Attention forward and backward, including packed-QKV and cuDNN/Triton
  alternatives where practical.
- LayerNorm forward, fused residual+LayerNorm, and LayerNorm backward.
- Classifier / softmax / cross-entropy / dlogits.
- AdamW, global norm, encoder, memsets, copies, and any profiler-visible
  runtime overheads.

Stacks to test per family when feasible:

- ThunderKittens 2.0 native kernels.
- cuBLAS and cuBLASLt, including epilogues and heuristic/result selection.
- cuDNN attention where supported by shape and precision.
- Triton kernels for attention, normalization, elementwise fusion, and GEMM
  variants where compile/runtime overhead is justified.
- Torch kernels/operators where they provide a faster exact-shape backend route
  or a useful reference point; native Torch rows that do not expose trainer
  state such as LayerNorm mean/rstd must be recorded separately from
  stats-producing trainer-compatible variants.
- CuTeDSL kernels for GEMM or fused epilogue variants where it can generate
  Blackwell-appropriate code.
- Plain CUDA baselines for small kernels and correctness fallback paths.

Do not require every stack to support every kernel. If a stack cannot reasonably
represent a kernel, record it as not applicable with the reason.
Package import is not enough evidence for codegen stacks: CuTeDSL rows need a
kernel compile/parity probe before timing. If the installed CuTeDSL path rejects
the target architecture or shape, record the exact rejection in the round log
instead of treating the stack as benchmarked.

## Benchmark Protocol

Use `best_runs.md` as the live scoreboard. For every candidate, record:

- Kernel family and exact GPT-2 shape.
- Provider stack.
- Build flags and runtime flags.
- Correctness command and result.
- Microbenchmark command and result.
- End-to-end smoke command and result, when the candidate is promoted into the
  trainer.
- Commit hash or working-tree marker.
- Log path and artifact path.
- Decision: promoted, rejected, or needs more data.

Minimum correctness gates before timing a candidate:

```bash
make -j test_matmul test_attention test_layernorm \
    test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm \
    train_gpt2cu \
    DEVICE_ARCH=SM120 \
    NO_MULTI_GPU=1 \
    NO_USE_MPI=1

./test_matmul
./test_attention
./test_layernorm
./test_bias
./test_gelu
./test_fused_classifier
./test_encoder
./test_adamw
./test_global_norm
```

Add stack-specific correctness tests before selecting a non-existing stack as a
default. For example, Triton, Torch, and CuTeDSL candidates need their own parity checks
against the current CUDA/cuBLASLt/TK reference outputs before trainer promotion.

Microbenchmarks should isolate provider performance by shape. The existing
`bench_sm120_matmul` is the starting point for GEMM and fused GEMM epilogues,
including the MLP `fwd+GeLU`, MLP-projection backward `dInp+dGeLU`, and
accumulated dWeight `dW+accum` paths. Extend the same pattern for attention,
LayerNorm, and any fused epilogue or runtime-overhead candidates.

Use `scripts/run_sm120_optimization_round.sh` for repeatable SM120 rounds. It
builds the SM120 correctness and benchmark targets, runs `test_matmul`,
`test_attention`, `test_layernorm`, plus the runtime-family correctness smokes
`test_bias`, `test_gelu`, `test_fused_classifier`, `test_encoder`,
`test_adamw`, and `test_global_norm`, runs the SM120
matmul/attention/LayerNorm/runtime microbenchmarks, runs the TinyStories smoke
into a round-specific
`log124M/5090_S_<run-label>` directory, captures a summary under
`scratch/sm120_rounds/<run-label>/`, records optional backend stack feasibility
in `backend-stacks.json` / `backend-stacks.md`, writes
`round-manifest.json` / `round-manifest.md` with the run config, commit,
changed-path count, toolchain output, and SHA256 identity for the built
binaries, and removes bulky model/state checkpoint outputs from that round
directory after metrics are captured. Manifest validation requires every
expected smoke-test, benchmark, and trainer binary to exist and carry SHA256
evidence, so a round cannot pass with a stale or omitted executable. Set
`KEEP_CHECKPOINTS=1` only when the checkpoint artifacts themselves are required.
Required stack-probe validation
does not require every allowed stack to be installed, but it does require an
evidence row and next action for every allowed stack in this file:
ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, Triton, Torch, CuTeDSL, and Plain
CUDA, plus the GPU runtime evidence precondition. The same probe also writes a
per-family backend applicability matrix for every required objective family and
allowed stack. Each family/stack row must say whether that stack is the current
baseline, a candidate, a fallback, missing/blocked, or not applicable, and every
not-applicable row must include the reason. This is how the round records that
some stacks are not reasonable providers for pointwise, reduction, runtime, or
non-GEMM families without treating that as missing benchmark coverage.
The allowed-stack, required-family, runtime-kernel, correctness-target, and
manifest-binary lists live in `dev/sm120_objective_contract.py`; the probe,
manifest writer, and round validator all import that module so their evidence
requirements cannot drift independently.
The exact GPT-2 selection-row requirements also live in that contract: GEMM
shape/pass requirements, runtime shape requirements, and fixed attention,
LayerNorm, classifier, optimizer, norm, encoder, memset, and copy shapes are
shared by the round validator and current-selection audit.
The harness runs `dev/validate_sm120_round.py` after non-dry runs to enforce the
requested log set, parse benchmark/training metrics, verify checkpoint cleanup,
validate the manifest, and write `scoreboard-candidates.md` for review before
copying rows into `best_runs.md`. When benchmark logs are required, the
validator also checks objective-family coverage and fails the round if any
required timing family is absent: GEMM forward, fused forward+GELU, dInput,
fused dInput+dGELU, dWeight, accumulated dWeight, attention forward/backward,
LayerNorm forward/fused-residual/backward, classifier, AdamW, global norm,
encoder, memsets, and copies. It also checks the exact GPT-2 GEMM shape matrix:
qkv, attention projection, MLP projection, and LM-head for plain forward; MLP
up for fused forward+GELU; all five shapes for dInput, dWeight, and
accumulated dWeight; and MLP projection for fused dInput+dGELU. For each
required GEMM pass and shape, it also checks provider coverage for
ThunderKittens, cuBLASLt, and cuBLAS. The cuBLAS rows are explicit cuBLAS plus
the existing CUDA pointwise passes where cuBLAS cannot represent a fused
epilogue. When both benchmark logs and the stack-probe matrix are present, the
validator also checks baseline-provider coverage across all objective families:
each family marked with a baseline provider in `backend-stacks.json` must have a
parsed benchmark row from that provider. This prevents a round from satisfying
`attention_forward`, `layernorm_forward`, or runtime-family coverage with the
wrong stack label.
When the manifest records `run_python_stack_benchmarks=1`, the validator also
requires every optional Python-stack benchmark log emitted by the round harness:
Torch matmul, Torch attention, Torch classifier, Torch runtime, the combined
Triton/Torch LayerNorm log, Triton matmul/attention/classifier/runtime, cuDNN
attention, and CuTeDSL matmul feasibility. A round cannot claim optional stack
benchmarking while silently omitting any required Python-stack log.
For Torch specifically, the validator also writes and enforces a `Torch
Objective Benchmark Coverage` table. Python-stack rounds must contain Torch
timing rows for every exact GPT-2 objective row in the shared contract: all
required GEMM pass/shape rows, attention forward/backward, LayerNorm
forward/fused/backward, classifier loss and dlogits, AdamW, global norm,
encoder, required bias/reduction rows, and required memset/copy rows.
The same report writes a `Selected Backend Rows` section and
`selected-backends.json`, selecting the fastest observed stack for each exact
benchmark row. This is the operational rule for Torch in the optimization
track: use Torch where that selected row is actually faster for the stated
scope, and keep the row scoped as a Python or operator prototype unless it has
a trainer-callable integration and TinyStories smoke evidence. The JSON artifact
also records whether the selected row already has a trainer call path, plus
row-level provenance for timing, config, stack-probe, correctness, source run
label, and source commit evidence. The current-selection audit verifies that
those referenced artifacts exist; runtime primitives that have no route-specific
correctness smoke must carry an explicit correctness-evidence note instead of an
unexplained empty list. For
example, native Torch SDPA may be the selected row for already-separated Q/K/V
experiments, while the packed-QKV trainer path still needs the `TorchPacked`
comparison to beat the current TK row before promotion.
The validator also writes a `Promotion Backlog` section and
`promotion-candidates.json` for selected rows without a trainer call path. That
backlog is the priority list for turning Torch, Triton, or other Python/codegen
wins into trainer-callable routes; each row records the candidate class, measured
edge versus the next observed stack when available, and the gate needed before
promotion. Native/direct and codegen integration candidates come before library
integrations, layout rewrites, and reference/state gaps. For example, a Triton
operator row is closer to the current C++ trainer than a Torch row that would
require adding libtorch or an equivalent native replacement. Native Torch SDPA
over separated Q/K/V is a valid reference win, but it is a layout-rewrite
candidate until a trainer-shaped packed path beats packed TK or the trainer
activation layout changes. Treat the backlog as round-local evidence: before
implementing a row, refresh the selected stack and current trainer baseline in
the same session. If the refreshed baseline wins, reject the backlog row as stale
instead of adding an integration.

Regenerate and audit the project-wide current SM120 selection after refreshing
native or optional-stack rounds with:

```bash
python3 dev/write_sm120_current_selection.py \
    --json-out scratch/sm120_rounds/current-sm120-selection.json \
    --markdown-out scratch/sm120_rounds/current-sm120-selection.md

python3 dev/audit_sm120_optimization_goal.py \
    --json-out scratch/sm120_rounds/current-sm120-audit.json \
    --markdown-out scratch/sm120_rounds/current-sm120-audit.md
```

The selection writer enforces the optional-stack decision contract before it
writes the consolidated artifact: every selected optional-stack row without a
trainer call path must have a matching `promotion-candidates.json` row and an
inactive decision. It records the optional non-trainer selected-row and
promotion-row counts in `current-sm120-selection.json`, writes a
`Resolved Optional-Stack Decisions` table to the Markdown artifact, and the
audit checks those counts and the table against the optional round.
The audit is the host-side guard for the "use the fastest stack where it is a
real trainer route" rule: every selected optional-stack row without a trainer
call path, including Torch, Triton, cuDNN, CuTeDSL, and future optional stacks,
must have a matching promotion-candidate row and be represented in the
consolidated resolved-decision list with an inactive/reference decision before
the current trainer mix is reported. Torch additionally keeps explicit
benchmark-log presence checks for its matmul, attention, classifier, runtime,
and LayerNorm comparison rows, and verifies that the optional-stack round's
scoreboard records all `43/43` Torch objective benchmark rows from
`dev/sm120_objective_contract.py`. The audit also
checks that the current native trainer mix covers every objective family from
`dev/sm120_objective_contract.py`, so the selection artifact cannot pass while
omitting a GPT-2 operator family. The same audit checks exact selected-row
coverage for the required GPT-2 shape matrix: all GEMM pass/shape rows,
attention forward/backward, LayerNorm forward/fused/backward, classifier loss
and dlogits, AdamW, global norm, encoder, required bias/reduction rows, and
required memset/copy rows. It also replays the stack-probe contract for both
audited source rounds: every objective stack plus GPU runtime must have status,
evidence, candidate-use, and next-action text, and every family/stack row must
have status, reason, and next action. The audit also checks each source round's
manifest for SM120 config, git/toolchain metadata, expected smoke/benchmark/
trainer binary rows, and SHA256 evidence; optional-stack rounds must record
`run_python_stack_benchmarks=1`.
For one-command replay from the round harness, set
`RUN_CURRENT_SELECTION_AUDIT=1`; override `SM120_SELECTION_NATIVE_ROUND` and
`SM120_SELECTION_OPTIONAL_ROUND` when auditing freshly generated native or
optional-stack evidence instead of the current default artifact pair. Use
`DRY_RUN=1` first when checking the harness command expansion without building
or running a new round.
Resolved decisions are tracked in `dev/sm120_promotion_decisions.json`. The
validator loads that registry by default, annotates candidate JSON with
`decision_status`, `decision_active`, `decision`, and `decision_evidence`, and
writes an `active_promotion_candidates` list for rows that still need
implementation attention. Keep rejected same-session refreshes, operator-only
Torch rows, layout-only attention rows, and non-trainer-shape rows in the
registry instead of deleting their benchmark evidence.
This is especially important for library integrations: adding libtorch or a
native replacement is only justified when the refreshed same-session comparison
still beats the current C++ route by enough to survive parity, build, and
TinyStories smoke gates.
Rows that do not correspond to an actual GPT-2 trainer shape must stay as
operator evidence. For LayerNorm and fused residual LayerNorm in GPT-2 124M,
the trainer shape is hidden width `C=768`; wider `C=3072` rows are useful stress
tests but are not trainer-promotion candidates.

Promotion rules:

- Correctness passes first.
- Microbenchmarks must beat the current provider for the target shape.
- A trainer-integrated candidate must improve or preserve the final TinyStories
  smoke. A local row win is not enough.
- Prefer a mixed selector over one global backend flag.
- Reject and document candidates that fail compile, smoke, parity, or end-to-end
  timing.

## Final Smoke Gate

The final candidate mix must run the TinyStories GPT-2 smoke from `new-goal.md`.
For quick validation, keep `-x 3`; for stability evidence, rerun with a longer
cap such as `-x 10` after the 3-step gate passes.

```bash
./train_gpt2cu \
    -i "dev/data/tinystories/TinyStories_train.bin" \
    -j "dev/data/tinystories/TinyStories_val.bin" \
    -o "log124M/5090_S" \
    -v 250 -s 20000 -g 144 \
    -h 0 \
    -b 64 -t 1024 -d 524288 \
    -r 0 \
    -z 1 \
    -c 0.1 \
    -l 0.0006  -q 0.0 -u 700 -n 5000 \
    -y 0  \
    -e "d12" \
    -x 3
```

A promoted result must report:

- Build command and backend flags.
- Step timings.
- Throughput.
- Loss/norm sanity compared with the baseline.
- Whether `use_master_weights` and `gelu_fusion` match the intended config.
- Exact `best_runs.md` rows updated.

Target to beat:

- llm.c baseline: about `2680 ms` steady-state from the supplied 3-step sample.
- Restored llm.kittens SM120 baseline: about `2508 ms`.
- Current best noted in `new-goal.md`: about `2469 ms`.

Use same-machine reruns for final comparison. Do not compare a new result
against an old run if clocks, driver, CUDA version, build flags, or model-output
cleanup changed.

## Disk And Evidence Rules

- Preserve `log124M/5090_S`.
- Keep `main.log`, `run.log` when present, and `DONE_*` markers needed to prove
  a run completed.
- Frequently remove bulky `model_*.bin` and `state_*.bin` outputs after their
  metrics are captured, because they fill the disk quickly.
- Do not delete evidence from other candidate directories unless the cleanup is
  explicitly part of the benchmark workflow and the useful metrics have already
  been copied into `best_runs.md`.

## Agent Checklist

When starting this goal:

- Read `new-goal.md`, this file, `best_runs.md`, `docs/sm120-rtx5090-baseline.md`,
  `goal.md`, `docs/kernel-reference.md`, and the recent SM120 entries in
  `CHANGELOG.md`.
- Check the current worktree before editing. It commonly contains useful runtime
  evidence and untracked benchmark artifacts.
- Start with SM120 because it is locally testable.
- Benchmark stacks separately before mixing them.
- Promote only shape-specific winners.
- Keep `best_runs.md` current after every accepted or rejected experiment.
- End each optimization round with a concrete decision and the next highest
  expected-payoff kernel family.
