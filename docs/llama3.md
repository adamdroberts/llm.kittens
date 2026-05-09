# Llama-3 (M6 + M7)

Llama-3 is a sibling entrypoint, `train_llama3.cu`, that shares `llmc/` with
`train_gpt2.cu`. The Python converter path, RoPE, RMSNorm, SwiGLU, GQA with TK
forward plus supported-shape TK backward paths, dataloaders, and the C++ trainer
loop have landed. RoPE is folded into GQA Q/K materialization and
packed-gradient unpermute for fallback shapes, and TK-supported shapes now
rotate Q/K in the tile-load path. H100 runtime validation remains the core
M6/M7 work.

The full TODO is in [`../goal.md`](../goal.md#m6--llama-3-1b-single-node--partial).

## What changes vs GPT-2

| Component | GPT-2 | Llama-3 |
|---|---|---|
| LayerNorm | LayerNorm (mean + variance + bias) | RMSNorm (variance only, no bias) |
| Position encoding | Learned `wpe` embedding | Rotary (RoPE), applied to Q/K |
| Attention | Multi-head attention (MHA) | Grouped-query attention (GQA), with RoPE fused into tile loads for supported shapes |
| MLP | `up → GELU → proj` | `silu(gate) * up → proj` (SwiGLU) |
| Bias terms | yes (matmul + LN) | no (none anywhere) |
| LM head | tied with embedding | untied |
| Tokenizer | GPT-2 BPE, vocab 50257 | Llama-3 BPE, vocab 128256 |
| Header magic | `20240326` v3/v5 | `20240803` v5 |

## New kernels

All four sit alongside their GPT-2 counterparts in `llmc/`:

- **`llmc/rmsnorm.cuh` + `llmc/tk/rmsnorm_tk.cuh`** — RMSNorm forward /
  backward + fused-residual variant. Forward and fused-residual forward use the
  TK fork for supported widths; backward is a plain CUDA correctness baseline.
  Landed, integrated into `train_llama3.cu`, and covered by the compile-ready
  `make test_rmsnorm` CPU-reference smoke harness. Runtime validation remains
  pending.
- **`llmc/rope.cuh` + `llmc/tk/rope_tk.cuh`** — wraps
  `ThunderKittens/kernels/rotary/rotary.cu`. Backward is the inverse rotation
  with `sin -> -sin`. Landed, integrated into `train_llama3.cu`, and covered
  by the compile-ready `make test_rope` CPU-reference smoke harness; numerical
  runtime validation remains pending.
- **`llmc/attention_gqa.cuh` + `llmc/tk/attention_gqa_h100.cuh`** — the
  highest-risk component in the project. TK causal forward and supported-shape
  TK backward paths have landed for BF16 H100 shapes, with slow CUDA fallback
  for unsupported shapes. The remaining target kernel work should extend the
  MHA template:
  - Accept `n_q_heads`, `n_kv_heads`, `n_rep = n_q / n_kv`.
  - Forward: repeat KV across query groups *inside* the kernel (landed for
    supported shapes; no host-side replication).
  - Backward: accumulate `dK`/`dV` into KV-head gradient buffers across the
    `n_rep` Q heads (landed for supported shapes; H100 validation pending).
  - RoPE: supported TK shapes rotate Q/K inside the tile-load path; fallback
    shapes keep the fused materialization/unpermute path.
  `dev/validate_attention_gqa_reference.py` validates the matching CPU-only
  PyTorch reference math for `B=1 T=128` and `B=1 T=256`, including backward
  gradients into packed Q/K/V. `scripts/validate_goal_h100.sh gqa-runtime` runs
  that reference plus the CUDA/TK `test_attention_gqa` comparison on H100.
- **`llmc/swiglu.cuh`** — plain CUDA forward/backward
  (`out = silu(gate) * up`). No TK benefit. Landed and covered by the
  compile-ready `make test_swiglu` CPU-reference smoke harness.

## Model spec (`LlamaConfig`)

Per `llm.c/train_llama3.py:245-262`:

| Variant | Layers | Heads (Q / KV) | Hidden | FFN hidden | Context |
|---|---|---|---|---|---|
| `llama3:1B` | 16 | 16 / 8 | 2048 | 8192 (~5632 effective with SwiGLU) | 2048 |
| `llama3:8B` / `llama3.1:8B` | 32 | 32 / 8 | 4096 | 14336 | 2048 (8192 for 3.1) |

`ParameterTensors` differs from GPT-2:

- No biases anywhere.
- Three MLP weights per layer: `fcw_up`, `fcw_gate`, `fcprojw`. This matches
  the checkpoint writer order: Python `c_fc` / Meta `w3` first, then Python
  `c_fc2` / Meta `w1` gate.
- Untied `lm_head` (separate from input embedding).
- RMSNorm scales only — no bias.

Port the layout from `llm.c/train_llama3.py:245-262` and the descriptor parser
from `llm.c/train_llama3.py:879` (header magic `20240803` v5). The C++ surface
also validates that a checkpoint's payload byte count matches the parsed
parameter layout before CUDA startup.

`train_llama3.cu` currently implements the config/parser/parameter-layout
surface plus the forward/backward/update loop. Running it with default
`-x 0` is a dry run that prints the resolved config and validates checkpoint
payload size without touching CUDA. Passing `-x >0` trains with TK GQA forward
and supported-shape TK backward where available, with slow CUDA fallback for
unsupported shapes.

Checkpointing mirrors the GPT-2 trainer: rank 0 writes `model_%08d.bin`, every
rank writes `state_%08d_%05d.bin`, and rank 0 publishes `DONE_%08d` after the
barrier. The state file stores AdamW moments, optional FP32 master weights, RNG
state, and dataloader cursor/shuffle state. `-y 1` resumes from the highest
completed `DONE_*` step in `-o OUTPUT_DIR`; this path is compile-checked, but a
real restart still needs H100 runtime validation.
The Llama trainer also writes the same rank-0 `OUTPUT_DIR/main.log` format as
the GPT-2 trainer: `tel` validation loss, `eval` HellaSwag/eval accuracy, and
`trl` train loss with learning rate and gradient norm. The H100 harness uses
`dev/validate_training_log.py` to check those metrics after stability and full
runs.

## Checkpoint converter

`train_llama3.py` (in repo root) ports `from_pretrained_llama3_hf` and
`write_model` from `llm.c/train_llama3.py`. This produces
`llama3.1_8B_bf16.bin` from HuggingFace `meta-llama/Meta-Llama-3.1-8B`.
Requires `huggingface-cli login` because the Llama-3 weights are gated.
The writer now emits the same hidden-dim header field as the C++ checkpoint
writer. `dev/validate_llama3_converter.py` exercises `write_model` directly on a
tiny deterministic model, checks the header and BF16 payload tensor order, and
can dry-parse the result with `train_llama3cu -x 0 -z 2`.

`dev/download_llama3.py` is the user-facing wrapper:

```bash
huggingface-cli login
python dev/download_llama3.py llama3.1:8B    # → llama3.1_8B_bf16.bin
python dev/validate_llama3_converter.py --cpp-validate --train-binary ./train_llama3cu
python dev/download_llama3.py --validate-only llama3.1_8B_bf16.bin --cpp-validate
python dev/download_llama3.py --write-synthetic-checkpoint /tmp/llama_synthetic.bin --cpp-validate
```

Post-write validation checks the magic/version, parameter payload byte count,
and hidden-dim metadata. `--cpp-validate` additionally runs
`train_llama3cu -e CHECKPOINT -x 0`, which parses the checkpoint and validates
payload size without initializing CUDA. Add `--cpp-zero-stage` and
`--cpp-processes` to run that C++ parser through a host-only ZeRO layout path in
the same command.
`--write-synthetic-checkpoint` writes a tiny deterministic BF16 checkpoint with
valid Llama metadata, useful for parser and ZeRO layout smoke tests when gated
HF weights are unavailable.
The H100 harness can run that C++ parser later without re-downloading weights:

```bash
LLAMA_DRY_CHECKPOINT=llama3.1_8B_bf16.bin scripts/validate_goal_h100.sh llama-dry
scripts/validate_goal_h100.sh llama-converter-smoke
scripts/validate_goal_h100.sh llama-checkpoint-smoke
scripts/validate_goal_h100.sh llama8b-convert
```

`llama8b-convert` is the real gated HF conversion gate. It validates
`${LLAMA8B_CHECKPOINT:-./llama3.1_8B_bf16.bin}` when present, otherwise it
converts `${LLAMA8B_MODEL:-llama3.1:8B}` into `${LLAMA8B_OUTPUT_DIR:-.}`. In
both cases it runs `--cpp-validate` through the default ZeRO-2/16-process dry
layout expected by the 2-node 8×H100 target.

## Datasets

Llama-3 shards are not interchangeable with GPT-2 shards (different vocab,
different token width). Pass `--model_desc llama-3` to every dataset prep
script:

```bash
python dev/data/tinyshakespeare.py --model_desc llama-3
python dev/data/tinystories.py     --model_desc llama-3
python dev/data/fineweb.py         --model_desc llama-3
python dev/data/hellaswag.py       --model_desc llama-3
```

Header magic `20240801` v7, **uint32** tokens, vocab 128256. The training
`DataLoader` dispatches on that header magic and still accepts GPT-2 uint16
shards for GPT runs.

HellaSwag uses a separate eval-file format: GPT-2 remains
`hellaswag_val.bin` with magic `20240522` v1 and uint16 tokens; Llama-3 writes
`hellaswag_val_llama3.bin` with magic `20240802` v7 and uint32 tokens.
`EvalLoader` consumes both formats, and `train_llama3.cu` routes Llama HellaSwag
eval to `hellaswag_val_llama3.bin`.

`python dev/validate_data_artifacts.py` validates prepared GPT-2 and Llama
training/eval files without CUDA. It checks the Llama uint32 magic/version,
exact payload size, sampled token ranges, and the full HellaSwag eval stream
when those files exist.
`make test_dataloader && ./test_dataloader` covers the same GPT-2/Llama dispatch
path through the C++ `DataLoader` and `EvalLoader` using synthetic files.

## Single-node target (M6) — `scripts/run_llama3_1B.sh`

8×H100 ZeRO-1, B=32 T=2048, LR=3e-4, warmup=2000, cosine→0.1, ~250 ms/step,
~36 hours for 30 B tokens on Llama-tokenized FineWeb-edu 100B. The script is
present and syntax-checked; runtime awaits H100 validation and TK GQA numerical
validation. `scripts/validate_goal_h100.sh llama1b-stability` is the bounded
1000-step version of this gate. After the run it validates
`log_goal_llama1b_stability/main.log`, requiring final validation/train metrics,
train-loss decrease, and final HellaSwag/eval metrics unless
`LLAMA1B_STABILITY_HELLASWAG=0`.

Why 1B and not 8B for the single-node target: 8B exceeds 80 GB at ZeRO-1
because optimizer state (FP32 m+v+master) alone is ~96 GB. ZeRO-2 fixes this
and is compile-wired through the reduce-scatter/sharded-optimizer path. ZeRO-3
now owns local BF16 parameter shards and all-gathers into the current full
compute layout; H100/NCCL validation is still pending.

## Multi-node target (M7) — `scripts/multi_node/run_llama3_8B_fs.sbatch`

2 nodes × 8×H100, ZeRO-2 across the 16 ranks, FS rendezvous. The Slurm script
mirrors `run_gpt2_124M_fs.sbatch` (M5) plus Llama-specific defaults, but it is
still pending real H100/NCCL validation. It accepts `MODEL_DESC=llama3.1:8B
SEQ_LEN=8192` for the longer-context variant.

## Validation plan

1. **GQA forward/backward** — compare against the CPU-reference smoke harness
   and PyTorch. `dev/cuda/test_attention_gqa.cu` covers `B=1 T=128` for TK
   forward plus fallback backward and `B=1 T=256` for supported-shape TK
   backward. `dev/validate_attention_gqa_reference.py` covers the CPU-only
   PyTorch reference equivalence for both shapes; `scripts/validate_goal_h100.sh
   gqa-runtime` is the H100 TK comparison gate.
2. **Tile-load RoPE validation** — re-validate the supported-shape tile-load
   path against the materialized fallback path and a PyTorch reference.
3. **Checkpoint/restart** — run a short `run_llama3_1B.sh` job with `-n`, then
   restart with `-y 1` and confirm loss/RNG/dataloader continuity. The harness
   checks the `DONE_*`, model, and rank-0 state files for the initial and final
   checkpoint steps with `dev/validate_llama_checkpoint_artifacts.py`, then
   validates the resumed `main.log`.
4. **End-to-end Llama-3 1B** — run `scripts/validate_goal_h100.sh
   llama1b-stability` for 1000 steps on FineWeb-edu-100B. Loss decreases
   overall, no NaNs; HellaSwag accuracy tracks expected curve. The log verifier
   checks final validation/train/eval evidence and optional
   `LLAMA1B_STABILITY_MAX_VAL_LOSS` / `LLAMA1B_STABILITY_MIN_HELLASWAG`
   thresholds. The longer `run_llama3_1B.sh` target continues from there toward
   the full run, and `scripts/validate_goal_h100.sh llama1b-full` applies the
   same `main.log` checks at the full-run final step.

See [testing.md](testing.md) for the rest of the test pyramid.
