# CLI reference

Every command-line flag exposed by the trainers and the supporting helpers,
grouped by what they affect. Defaults match the in-source `error_usage()`
listings in [`train_gpt2.cu`](../train_gpt2.cu) and
[`train_llama3.cu`](../train_llama3.cu); when those drift, source wins.

The companion validation harness has its own page —
[`validation-harness.md`](validation-harness.md).

## `train_gpt2cu`

Builds with `make train_gpt2cu`. Single binary that handles GPT-2 and the
GPT-3 descriptor variants (`gpt3:c384` … `gpt3:c12288`).

### Model and data

| Flag | Default | Purpose |
|---|---|---|
| `-e <model_or_bin>` | `gpt2_124M_bf16.bin` | BF16 checkpoint or descriptor (`gpt2:124M`, `gpt3:c768`, …). Selects the model dimensions. |
| `-i <pattern>` | `dev/data/tinyshakespeare/tiny_shakespeare_train.bin` | Training shard glob. GPT-2 shards are uint16 / magic `20240520` v1. |
| `-j <pattern>` | `dev/data/tinyshakespeare/tiny_shakespeare_val.bin` | Validation shard glob. |
| `-h <0\|1>` | `0` | Run HellaSwag eval. Requires `hellaswag_val.bin`. |

### Optimization shape

| Flag | Default | Purpose |
|---|---|---|
| `-b <int>` | `4` | Per-GPU micro batch size `B`. |
| `-t <int>` | `1024` | Sequence length `T`. GPT-3 scripts use `2048`. |
| `-d <int>` | `B*T*num_processes` | Total batch size in tokens. Sets the gradient-accumulation factor. |
| `-x <int>` | `-1` (one epoch) | Maximum optimizer steps. `0` triggers the host-only dry run that validates the descriptor / checkpoint header without CUDA. |

### Learning rate, regularization, stability

| Flag | Default | Purpose |
|---|---|---|
| `-l <float>` | `3e-4` | Peak learning rate. |
| `-u <int>` | `0` | Warmup iterations before the schedule starts. |
| `-q <float>` | `1.0` | Final LR fraction at end of training. `0.1` for Llama-style cosine→0.1. |
| `-c <float>` | `0.0` | Weight decay. |
| `-k <name>` | `cosine` | LR schedule name (`cosine`, `linear`). |
| `-sl <float>` | `0.0` (off) | Skip optimizer step if loss z-score exceeds this. |
| `-sg <float>` | `0.0` (off) | Skip optimizer step if gradient-norm z-score exceeds this. |
| `-w <0\|1>` | `1` | Keep FP32 master weights. |

### Evaluation and sampling

| Flag | Default | Purpose |
|---|---|---|
| `-v <int>` | `20` | Validation every `N` steps. |
| `-m <int>` | `20` | Max validation batches per evaluation. |
| `-s <int>` | `20` | Sample every `N` steps. |
| `-g <int>` | `64` | Tokens per sampling pass. |

### Output and checkpointing

| Flag | Default | Purpose |
|---|---|---|
| `-o <dir>` | `NULL` | Output log directory. When set, rank 0 writes `main.log`, model checkpoints, and rank state files. |
| `-n <int>` | `0` | Write a checkpoint every `N` steps. `0` disables checkpointing. |
| `-nk <int>` | `0` | Maximum checkpoint history (`0` = keep all). |
| `-nm <int>` | `0` | Major-checkpoint cadence (never deleted). |
| `-y <0\|1>` | `0` | Resume from the latest `DONE_*` step in `-o`. Requires `-o`. |

### Numerics, precision, recompute

| Flag | Default | Purpose |
|---|---|---|
| `-f <0\|1>` | `1` | TF32 override. |
| `-ge <0\|1\|2>` | per-GPU default | GELU fusion: `0` plain CUDA GELU, `1` opt-in TK MLP-up bias+GELU epilogue, `2` reserved. The dry run defaults `-1` to `0`; H100 numerical validation is still pending — see [`kernel-reference.md`](kernel-reference.md). |
| `-r <0\|1\|2>` | `1` | Recompute during backward: `0` none, `1` GELU, `2` GELU+LN. |
| `-a <0\|1>` | `0` | Overfit a single batch (debug only). |
| `-lg <int>` | `-1` | Log GPU memory/util every `N` steps. `-1` disables. |

### ZeRO and multi-process

| Flag | Default | Purpose |
|---|---|---|
| `-z <0..3>` | `0` | ZeRO stage. `0` data-parallel, `1` shard optimizer, `2` shard optimizer + grads, `3` shard parameters. See [`multi-gpu.md`](multi-gpu.md). |
| `-pn <int>` | `1` | Total number of processes (set by launcher). |
| `-pr <int>` | `0` | This process's rank (set by launcher). |
| `-pg <int>` | `8` | GPUs per node. |
| `-pi <method>` | `mpi` | NCCL init: `mpi`, `tcp`, `fs`. |
| `-ps <ip>` | `""` | Server IP for `tcp` init. |
| `-pp <path>` | `""` | Filesystem path for `fs` init (named `-pf` in source; both names are accepted via the same option byte). |

### Dry-run mode

`-x 0` is a host-only dry run: parses the descriptor or `.bin` header, fills
the parameter metadata, and runs the host-only ZeRO layout validator. CUDA is
not initialized. The harness uses this in `gpt-dry`.

### Logging output

When `-o <dir>` is set, rank 0 appends to `<dir>/main.log` in three line
formats parsed by [`dev/validate_training_log.py`](../dev/validate_training_log.py):

- `s:<step> tel:<loss>` — validation loss
- `s:<step> eval:<accuracy>` — HellaSwag / eval accuracy
- `s:<step> trl:<loss> lr:<lr> norm:<grad_norm>` — train step

Long-run launch scripts also write `run.log` with the launch metadata the
completion harness asserts.

## `train_llama3cu`

Builds with `make train_llama3cu`. Same CLI shape as `train_gpt2cu`, with the
following deltas. Llama-specific defaults differ (`T=2048`, no GPT-3
descriptors, no `-ge`/`-f`/`-a`/`-lg`/`-w` flags).

| Flag | Default | Purpose |
|---|---|---|
| `-e <model>` | `llama3:1B` | Model descriptor (`llama3:1B`, `llama3:8B`, `llama3.1:8B`) or BF16 checkpoint. |
| `-t <int>` | `2048` | Sequence length default. `8192` is supported for `llama3.1:8B`. |
| `-i / -j` | tinyshakespeare | Llama shards are uint32 / magic `20240801` v7; `dataloader.h` dispatches by header magic. |
| `-h <0\|1>` | `0` | Llama HellaSwag eval expects `hellaswag_val_llama3.bin` (uint32 / magic `20240802` v7). |
| `-z`, `-pn`, `-pr`, `-pg`, `-pi`, `-pf`, `-ps` | mirrors GPT-2 | ZeRO + multi-process flags identical. |
| `-y <0\|1>` | `0` | Resume from latest `DONE_*` in `-o`. The state file carries AdamW moments, optional FP32 master weights, RNG, and dataloader cursor. |

Flags absent from the Llama trainer because Llama has no biases, no learned
position embedding, and no opt-in epilogue today: `-ge`, `-f`, `-a`, `-lg`,
`-w`.

The Llama dry run is `-x 0` (host-only descriptor parse + ZeRO layout
validation), used by `scripts/validate_goal_h100.sh llama-dry`. The same path
also accepts `LLAMA_DRY_CHECKPOINT=...` for a header/payload check on a
converted gated HF checkpoint.

## Helper scripts

### `dev/download_starter_pack.sh`

Fetches `gpt2_tokenizer.bin`, `gpt2_124M.bin`, `gpt2_124M_bf16.bin`,
`gpt2_124M_debug_state.bin` from Karpathy's HF mirror (~600 MB). No flags.

### `dev/data/*.py`

Shared options across `tinyshakespeare.py`, `tinystories.py`, `fineweb.py`,
`hellaswag.py`, `mmlu.py`:

- `--model_desc gpt-2` (default) or `--model_desc llama-3` switches token width
  (uint16 vs uint32) and the header magic. Llama HellaSwag writes
  `hellaswag_val_llama3.bin`.
- Dataset-specific flags such as `fineweb.py --version edu --shard_size`
  follow `llm.c/dev/data/`.

### `dev/validate_data_artifacts.py`

```text
python dev/validate_data_artifacts.py [--self-test] [--full-token-scan]
                                      [--data-dir DIR]
```

Validates exact file sizes, magic/version pairs, token width, sampled
train-token ranges, and the full HellaSwag-style eval stream of every prepared
GPT-2/Llama train/eval `.bin` it finds. `--self-test` writes synthetic
artifacts to `/tmp` and asserts both pass and fail paths. The harness phase is
`data-artifacts`.

### `dev/validate_gpt2_starter_pack.py`

```text
python dev/validate_gpt2_starter_pack.py [--self-test]
```

Validates the four GPT-2 starter-pack artifacts without CUDA. The harness
phase is `starter-pack`.

### `dev/download_llama3.py`

```text
python dev/download_llama3.py llama3.1:8B [--output_dir .]
                              [--validate-only PATH]
                              [--write-synthetic-checkpoint PATH]
                              [--cpp-validate] [--train-binary BIN]
                              [--cpp-zero-stage N] [--cpp-processes N]
```

Wraps the gated HF Llama-3 conversion: requires `huggingface-cli login`. After
write, validates the BF16 header, payload size, and hidden-dim metadata.
`--cpp-validate` runs `train_llama3cu -e CHECKPOINT -x 0` with the requested
ZeRO layout. `--write-synthetic-checkpoint` generates a tiny deterministic
checkpoint useful when gated weights are unavailable.

### `dev/validate_llama3_converter.py`

```text
python dev/validate_llama3_converter.py [--cpp-validate]
                                        [--train-binary BIN]
```

Builds a tiny `LLaMA` model, fills each named parameter with a distinct BF16
value, runs `write_model`, checks the header and payload tensor order, and
optionally dry-parses the file with the C++ trainer. The harness phase is
`llama-converter-smoke`.

### `dev/validate_llama_checkpoint_artifacts.py`

```text
python dev/validate_llama_checkpoint_artifacts.py [--self-test]
                                                  --output-dir DIR
                                                  --steps S1[,S2]
                                                  --num-processes N
```

Parses `DONE_*`, `model_*.bin`, and `state_*_*.bin` headers and asserts
expected magic/version, step, rank, and process count. Used by `llama-resume`
and `llama8b-full`.

### `dev/validate_training_log.py`

```text
python dev/validate_training_log.py LOG [--require-decrease]
                                        [--final-step STEP]
                                        [--require-final-train]
                                        [--require-final-val]
                                        [--require-final-eval]
                                        [--max-val-loss V]
                                        [--min-eval-acc A]
                                        [--expected-val-loss V --tol REL]
                                        [--expected-eval-acc A --tol REL]
```

Validates rank-0 `main.log`. Used by every training phase
(`gpt2-smoke`, `llama-resume`, `llama1b-stability`, `gpt2-full`,
`llama1b-full`, `gpt2-two-node`, `llama8b-full`).

### `dev/compare_training_logs.py`

```text
python dev/compare_training_logs.py REF.log CAND.log [--limit STEPS]
                                                     [--rel-tol R] [--abs-tol A]
                                                     [--require-decrease]
```

First-`N`-step train-loss-curve comparator. Backs the `gpt2-two-node` phase.

### `profile_gpt2cu.py`

```text
python profile_gpt2cu.py [--gelu-fusion 0|1]
                         [--min-tensor-util 70]
                         [--output PREFIX]
                         [--report PATH]
                         [--csv-input PATH]
                         [--skip-build] [--skip-run]
```

Builds, runs, and post-processes Nsight Compute profiling for the
`profile_gpt2cu` binary. Defaults to fail when averaged MMA tensor-core
utilization is below 70%. `--csv-input` validates an existing raw CSV without
invoking `ncu`. The harness phase is `profile`.

### Source-guard validators

Most live under `dev/`. Each is run by
`scripts/validate_goal_h100.sh source-guards`; running them directly takes no
arguments and exits non-zero on a contract break:

| Script | Contract |
|---|---|
| `validate_build_contracts.py` | Makefile/BF16/Hopper+Blackwell/TK build invariants |
| `validate_epilogue_source.py` | TK GEMM bias+GELU epilogue + `-ge 1` plumbing |
| `validate_gqa_source.py` | GQA/RoPE tile-load routing and head mapping |
| `validate_runtime_markers.py` | Runtime success-marker contracts |
| `validate_training_source.py` | Rank-0 `main.log` evidence contracts |
| `validate_profile_source.py` | Nsight Compute profile-gate contracts |
| `validate_llama_conversion_source.py` | Gated HF conversion contracts |
| `validate_nccl_source.py` | Scalar NCCL counts + ZeRO-3 wiring |
| `validate_zero_layout.py` | ZeRO shard offset coverage |
| `validate_launch_scripts.py` | Launch-script `MAX_STEPS`/`DONE_*` alignment |
| `validate_goal_harness_coverage.py` | `goal.md` ↔ harness coverage |

Each is documented inline at the top of the file.
