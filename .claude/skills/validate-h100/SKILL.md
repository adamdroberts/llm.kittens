---
name: validate-h100
description: Drive the llm.kittens H100 validation harness — pick the right phase, set the required env vars, replay captured evidence, and avoid `goal-complete` mistakes. Use whenever the user asks to validate the goal, run a phase of `scripts/validate_goal_h100.sh`, replay an existing log, or check whether a milestone gate would pass.
---

# validate-h100

Use this skill when the user wants to **run, replay, or reason about** the
[`scripts/validate_goal_h100.sh`](../../../scripts/validate_goal_h100.sh)
harness. Do not use it for editing kernels or trainer code — that is the
[`llm-kittens-port`](../llm-kittens-port/SKILL.md) skill's job.

## Step 0 — read these in order

1. [`goal.md`](../../../goal.md) — find the unchecked `- [ ]` items the user
   needs evidence for.
2. [`docs/validation-harness.md`](../../../docs/validation-harness.md) — phase
   catalogue, validate-only modes, threshold table, recipes.
3. [`docs/testing.md`](../../../docs/testing.md) — test pyramid context.
4. [`docs/cli-reference.md`](../../../docs/cli-reference.md) — trainer flags
   the long phases use under the hood.

## Hard rules

- **Default with no args is `goal-core`.** That is also the right baseline for
  "is this branch ready" questions on an H100 host.
- **`goal-complete` requires `ALLOW_FULL_GOAL_RUN=1` and refuses to run with
  `ALLOW_NON_H100=1`.** It launches multi-hour jobs.
- **`goal-complete` requires every threshold listed in
  [`validation-harness.md`](../../../docs/validation-harness.md#required-thresholds-goal-complete).**
  Use `goal-complete-prereqs` before launching to fail fast on missing values.
- **Do not weaken the H100 gate.** `ALLOW_NON_H100=1` is for dry compile/debug
  only; `rtx5090-device` is a separate device-test path that is **not** valid
  evidence for unchecked H100 items.
- **Validate-only modes need real evidence.** `*_VALIDATE_ONLY=1` plus the
  matching `*_LOG` / output dir. Synthetic stubs are rejected by
  `goal-replay-smoke`.

## Picking a phase

| User intent | Phase |
|---|---|
| "Is the branch buildable / are docs/source contracts intact?" | `host-core` |
| "Does the H100 box have CUDA/NCCL/MPI?" | `preflight` |
| "Are kernels numerically OK on this H100?" | `smoke gqa-runtime gpt2` |
| "Run the GPT-2 124M reproduction" | `gpt2-full` (long) |
| "Compare 1-node vs 2-node loss" | `gpt2-two-node` |
| "Run the 1B Llama stability gate" | `llama1b-stability` |
| "Convert / validate gated 8B HF weights" | `llama8b-convert` |
| "Profile the GEMM/attention path" | `profile` (both `PROFILE_GELU_FUSIONS=0 1` for completion evidence) |
| "Run everything that's left to close `goal.md`" | `goal-complete` (with thresholds set) |
| "Replay captured evidence on a CPU host" | `goal-complete-prereqs` plus the `*_VALIDATE_ONLY=1` set |

## Common patterns

### Local build host (no CUDA)

```bash
scripts/validate_goal_h100.sh host-core
```

### H100 box, smoke and short runtime gates only

```bash
scripts/validate_goal_h100.sh         # = goal-core
scripts/validate_goal_h100.sh gqa-runtime gpt2-smoke zero3-smoke
```

### Replay captured evidence

```bash
PROFILE_VALIDATE_ONLY=1 PROFILE_CSV_DIR=evidence/profile \
GPT2_SMOKE_VALIDATE_ONLY=1 GPT2_SMOKE_LOG=evidence/gpt2_smoke/main.log \
LLAMA1B_STABILITY_VALIDATE_ONLY=1 \
LLAMA1B_STABILITY_LOG=evidence/llama1b_stability/main.log \
… \
scripts/validate_goal_h100.sh goal-complete-prereqs
```

### Real `goal-complete` run

```bash
ALLOW_FULL_GOAL_RUN=1 \
GPT2_SMOKE_MAX_VAL_LOSS=8.0 \
ZERO3_SMOKE_MAX_VAL_LOSS=8.0 \
LLAMA_RESUME_MAX_VAL_LOSS=10.0 \
LLAMA1B_STABILITY_MAX_VAL_LOSS=4.5 LLAMA1B_STABILITY_MIN_HELLASWAG=0.30 \
GPT2_FULL_EXPECTED_VAL_LOSS=2.85 GPT2_FULL_EXPECTED_HELLASWAG=0.294 \
GPT2_TWO_NODE_REL_TOL=0.005 \
LLAMA1B_FULL_MAX_VAL_LOSS=2.5 LLAMA1B_FULL_MIN_HELLASWAG=0.45 \
LLAMA8B_FULL_MAX_VAL_LOSS=2.0 LLAMA8B_FULL_MIN_HELLASWAG=0.55 \
scripts/validate_goal_h100.sh goal-complete
```

(Substitute the actual thresholds you are gating against; do not invent numbers.)

## Diagnosing failures

- A phase fails fast if its **success marker** is missing, even when the binary
  exited 0. The marker list is in
  [`validation-harness.md`](../../../docs/validation-harness.md#phase-catalogue).
- `source-guards` failures usually mean a code edit broke a contract a
  `dev/validate_*_source.py` file enforces. Read the named guard's source — it
  prints exactly which line/file violated the contract.
- `goal-replay-smoke` failures mean a **change in the harness logic itself**
  (validate-only branch, threshold validator, prereq list). Update both the
  guard expectations and the underlying logic together.
- A `*_RUN_LOG` mismatch means the captured evidence is from an older script
  revision; rerun the gate on H100 instead of editing the validator.

## When *not* to use this skill

- Editing or porting kernels → use [`llm-kittens-port`](../llm-kittens-port/SKILL.md).
- Documentation refactors → use the `deep-documentation` skill.
- Asking "what is done vs left?" → read [`goal.md`](../../../goal.md). The
  harness does not own that question.

## Verification before reporting "done"

- The expected harness phase exits 0 **and** the explicit success marker shows
  up in stdout. Tail-grep for it before reporting success.
- If you ran a long phase, confirm `main.log` and `run.log` (and the final
  checkpoint marker / artifacts where applicable) actually exist and parse
  through `dev/validate_training_log.py` or
  `dev/validate_llama_checkpoint_artifacts.py`.
- For `goal-complete`, every threshold in the table above must have been set;
  otherwise `goal-complete-prereqs` would have rejected the launch.
