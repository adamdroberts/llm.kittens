# Build and run

Today the project builds **`make all`** end-to-end for the compile-ready GPT-2
path: `test_matmul`, `train_gpt2cu`, and `test_gpt2cu`. `make train_llama3cu`
also compiles the Llama trainer loop with checkpoint/resume, TK GQA forward,
and supported-shape backward where available, with CUDA fallback elsewhere.
TK-supported GQA shapes rotate Q/K inside the tile-load path, while fallback
shapes use fused materialization/unpermute. Runtime H100 parity is still pending — see
[`../goal.md`](../goal.md).

`DEVICE_ARCH=SM100`, `SM103`, and `SM120` are build-supported for Blackwell
with ThunderKittens 2.0. Hopper-only TK model kernels currently fall back to
plain CUDA correctness kernels on those targets; optimized B200/GB200 kernels
remain separate porting work.

## Toolchain

| Requirement | Why |
|---|---|
| CUDA Toolkit ≥ 12.4 | nvcc with `-std=c++20`; `sm_90a`, `sm_100a`, `sm_103a`, and `sm_120a` build targets |
| GCC ≥ 11 (or Clang ≥ 14) | host compiler for c++20 |
| ThunderKittens checkout | header-only; expected at `../ThunderKittens` by default |
| H100 (sm_90a) | primary optimized runtime target for goal evidence |
| Blackwell (sm_100a / sm_103a / sm_120a) | compile-supported; Hopper-only model kernels use CUDA correctness fallbacks pending optimized Blackwell kernels |
| NCCL ≥ 2.x + libnccl-dev | optional, multi-GPU |
| OpenMPI | optional, multi-node MPI init |
| Python ≥ 3.10 | only for `dev/data/*.py` preprocessing scripts |

`requirements.txt`: `tqdm`, `numpy<2`, `torch`, `tiktoken`, `transformers`, `datasets`, `requests`.

## TK_ROOT

The Makefile resolves the ThunderKittens checkout via `TK_ROOT`:

```bash
make                                  # default: TK_ROOT=$(abspath ../ThunderKittens)
make TK_ROOT=/path/to/ThunderKittens
```

If `$(TK_ROOT)/include/kittens.cuh` is missing the Makefile prints a warning
and the compile will fail at the first `#include <kittens.cuh>`.

## Targets

```bash
make                  # default = test_matmul + train_gpt2cu + test_gpt2cu
make test_matmul      # GEMM forward, opt-in bias+GELU epilogue, and dWeight smoke tests
make test_attention   # GPT MHA fwd/bwd smoke harness against CPU reference
make test_layernorm   # GPT LayerNorm fwd/fused/bwd smoke harness against CPU reference
make test_rope        # M6 RoPE fwd/bwd smoke harness against CPU reference
make test_rmsnorm     # M6 RMSNorm fwd/fused/bwd smoke harness against CPU reference
make test_swiglu      # M6 SwiGLU fwd/bwd smoke harness against CPU reference
make test_attention_gqa # M6 GQA + RoPE smoke harness against CPU reference
make train_gpt2cu     # GPT-2/GPT-3 trainer compile path
make gpt2_validate    # M2 forward-only GPT-2 loss gate against debug_state
make test_gpt2cu      # GPT-2 numerical parity test compile path
make train_llama3cu   # M6 Llama trainer/checkpoint-resume; TK GQA fwd/supported bwd path

# Separate optional target:
make profile_gpt2cu   # M8 profiling binary compile path

make clean            # removes binaries and build/*.o
```

The default invocation prints the build configuration banner before compiling:

```
---------------------------------------------
llm.kittens build configuration
---------------------------------------------
TK_ROOT          : /…/ThunderKittens
NVCC arch        : sm_90a
Precision        : BF16 (locked)
---------------------------------------------
✓ OpenMP found
✓ NCCL found, multi-GPU enabled
✓ MPI enabled
---------------------------------------------
```

## Build flags

The Makefile picks these defaults; override on the command line if needed.

| Flag | Default | Effect |
|---|---|---|
| `TK_ROOT=` | `$(abspath ../ThunderKittens)` | ThunderKittens checkout |
| `FORCE_NVCC_O=` | `3` | nvcc optimisation level |
| `DEVICE_ARCH=SM90` | `SM90` | CUDA/TK architecture; supported values are `SM90`, `SM100`, `SM103`, and `SM120` |
| `NO_OMP=1` | unset | Disable OpenMP host parallelism |
| `NO_MULTI_GPU=1` | unset | Disable NCCL even if installed |
| `NCCL_DIR=` | unset | Enable NCCL from a custom prefix containing `include/nccl.h` and `lib*/libnccl.so` |
| `NCCL_INCLUDE_PATH=` | derived from `NCCL_DIR` when set | Override NCCL headers when using a custom install |
| `NCCL_LIB_PATH=` | derived from `NCCL_DIR` when set | Override NCCL library path when using a custom install |
| `NO_USE_MPI=1` | unset | Disable OpenMPI even if installed |
| `OPENMPI_DIR=` | `/usr/lib/x86_64-linux-gnu/openmpi` | OpenMPI prefix |

The Makefile hard-codes:

- `--use_fast_math -std=c++20`
- `--expt-extended-lambda --expt-relaxed-constexpr`
- default `DEVICE_ARCH=SM90`: `-DKITTENS_SM90 -gencode arch=compute_90a,code=sm_90a`
- opt-in `DEVICE_ARCH=SM100`: `-DKITTENS_SM100 -gencode arch=compute_100a,code=sm_100a`
- opt-in `DEVICE_ARCH=SM103`: `-DKITTENS_SM103 -gencode arch=compute_103a,code=sm_103a`
- opt-in `DEVICE_ARCH=SM120`: `-DKITTENS_SM120 -gencode arch=compute_120a,code=sm_120a`
- `-DENABLE_BF16`
- `-I$(TK_ROOT)/include -I$(TK_ROOT)/prototype`

`PRECISION=FP16` and `PRECISION=FP32` are not supported in v1. The compile
errors out fast — see [precision.md](precision.md).

## GPU sniff

When `nvidia-smi` is available, the Makefile prints a warning if the lowest-CC
GPU does not match `DEVICE_ARCH`:

```
⚠ Detected GPU compute capability 86; DEVICE_ARCH=SM90 targets sm_90a.
  Build will still proceed.
```

Build still proceeds — useful when cross-compiling on a non-Hopper machine for
a Hopper deployment. Set `CI=true` to suppress the sniff.

This build-time warning is intentionally weaker than the goal harness. The
target-host `scripts/validate_goal_h100.sh preflight` and `cuda-runtime` probes
target H100 / sm90-class GPUs by default. For Blackwell compile checks, run:

```bash
scripts/validate_goal_h100.sh blackwell-compile
```

For datacenter Blackwell runtime probes, run:

```bash
scripts/validate_goal_h100.sh blackwell-device
```

For RTX 5090 device checks, run:

```bash
scripts/validate_goal_h100.sh rtx5090-device
```

The Blackwell/RTX probe phases skip NCCL/MPI by default. They do not replace
the unchecked H100 performance/parity evidence in `goal.md`.

## Smoke test

```bash
make test_matmul
./test_matmul

make test_attention
./test_attention

make test_layernorm
./test_layernorm
```

Expected output on an H100:

```
Device: NVIDIA H100 80GB HBM3 (sm_90)

──── 1024^3 square (default)  M=1024 N=1024 K=1024 ────
  max abs diff = 0.???  (tolerance 0.50)  PASS

──── GPT-2 124M MLP up (default)  M=4096 N=3072 K=768 ────
  max abs diff = 0.???  (tolerance 0.50)  PASS

──── GPT-2 124M LM head (small_n fallback)  M=4096 N=50304 K=768 ────
  max abs diff = 0.???  (tolerance 0.50)  PASS

──── forward bias+GELU epilogue (default)  M=1024 N=4096 K=1024 ────
  pre-GELU max abs diff = 0.????  GELU max abs diff = 0.????  (tolerance 0.50)  PASS

... dWeight overwrite and accumulated cases ...

──── 6/6 passed ────
```

Run on a non-H100 GPU prints a warning and continues. Blackwell builds use CUDA
fallbacks for the Hopper-only model kernels, so they are functional correctness
checks rather than performance evidence for the optimized TK path.

`test_attention` checks GPT-style packed Q/K/V attention against an independent
CPU reference. It covers direct TK forward with fallback backward (`T=192`) and
padded TK forward with supported-shape TK backward (`T=256`).

`test_layernorm` checks GPT LayerNorm forward, fused residual+LayerNorm forward,
saved `mean`/`rstd`, and backward accumulation into `dinp`, `dweight`, and
`dbias` against independent CPU references.

## Quickstart (compile-ready, runtime parity pending)

The intended end-to-end command for the smallest training path is unchanged
from `llm.c`:

```bash
# 1. Reference checkpoints + tokenizer for GPT-2 124M (4 files, ~600 MB).
./dev/download_starter_pack.sh

# 2. Tokenize tiny-shakespeare for a smoke run.
python dev/data/tinyshakespeare.py

# 3. Build.
make train_gpt2cu

# 4. Train 100 steps from the bf16 124M checkpoint.
./train_gpt2cu \
    -i dev/data/tinyshakespeare/tiny_shakespeare_train.bin \
    -j dev/data/tinyshakespeare/tiny_shakespeare_val.bin \
    -e gpt2_124M_bf16.bin \
    -b 4 -t 1024 -x 100 -v 20 -s 0
```

This compiles today, but a real run still needs H100 access, the downloaded
starter pack, and the remaining parity work tracked in [`../goal.md`](../goal.md).
Add `-ge 1` to opt GPT-2's MLP up-projection into the compile-wired TK
bias+GELU epilogue after validating numerics on your H100 host.
When `-o OUTPUT_DIR` is set, rank 0 writes `OUTPUT_DIR/main.log` with
validation loss (`tel`), eval accuracy (`eval`), and train loss/LR/grad-norm
(`trl`, `lr`, `norm`) entries. The long H100 harness phases use
`dev/validate_training_log.py` to turn that file into pass/fail evidence.

## Single-node and multi-node scripts (M4–M7)

The GPT-2/GPT-3 scripts under `scripts/` now mirror their `llm.c/scripts/`
counterparts, with repo-local paths and inline H100 NCCL defaults. The Llama-3
single-node and 8B filesystem-rendezvous scripts are also present. All launch
scripts syntax-check; distributed/runtime parity is still gated on H100/NCCL
runs and TK GQA numerical validation.
The full status table is in [multi-gpu.md](multi-gpu.md#per-script-status).

## H100 goal validation

On the target H100 machine, run the bounded validation harness before checking
off any remaining runtime gate in [`../goal.md`](../goal.md). The full phase
catalogue, env-var list, and validate-only recipes live in
[`validation-harness.md`](validation-harness.md); the summary below is the
quick-start.

```bash
scripts/validate_goal_h100.sh
```

The default `goal-core` phases check H100/CUDA/NCCL/MPI prerequisites, build
the compile targets, syntax-check launch/data shell scripts and Python
data/converter/profile helpers, run source-level CUDA/NCCL/ZeRO,
GQA/RoPE, BF16/Hopper+Blackwell/TK build, GELU-epilogue source, profile-gate source, Llama conversion source, rank-0 training-log evidence, launch-script, and runtime-marker contract guards, run
a tiny CUDA runtime/device-allocation probe, validate any prepared GPT-2/Llama training and HellaSwag `.bin`
artifacts found in the known data directories, run the host-only C++
DataLoader/EvalLoader smoke, run the CPU-only GQA/RoPE reference check, validate
the profile CSV parser/threshold path with synthetic data, validate the Llama
converter writer header/payload order with a tiny model, verify the GPT-2
starter-pack files, checkpoint metadata, tokenizer, and debug-state payload,
run kernel smoke tests, run GPT-2 forward/parity binaries, run GPT-2/GPT-3 and
Llama descriptor dry-runs with host-only 8-process ZeRO layout validation, write
and validate a tiny synthetic Llama checkpoint through the host-only parser,
validate host-only ZeRO-3 layouts, and check ZeRO request diagnostics.
On a local machine without a usable CUDA runtime, run
`scripts/validate_goal_h100.sh host-core` after building the binaries to execute
the non-CUDA-runtime subset of those gates against existing artifacts.
Set `GPT_DRY_CHECKPOINT=/path/to/model.bin` when running `gpt-dry` to include
GPT checkpoint header and payload-size validation without CUDA initialization.
The built-in GPT-3 dry-runs assert descriptor source, channel count, and ZeRO-2
layout output for every supported channel preset.
Set `LLAMA_DRY_CHECKPOINT=/path/to/llama.bin` when running `llama-dry` to run
the same host-only C++ checkpoint parser and payload-size validation for
converted Llama weights. The dry-run uses the same `set_zero_configs` helper as
runtime and reports the validated local shard parameter count; the built-in
descriptor dry-runs assert the Llama source and ZeRO layout markers.
Use `scripts/validate_goal_h100.sh llama-checkpoint-smoke` for a repeatable
local checkpoint parser smoke that does not need gated HF weights.
Use `scripts/validate_goal_h100.sh llama-converter-smoke` to validate
`train_llama3.py::write_model` itself without gated HF weights.
Use `scripts/validate_goal_h100.sh llama8b-convert` for the real gated HF
Llama-3.1 8B conversion/load gate; it validates an existing checkpoint or
converts `${LLAMA8B_MODEL:-llama3.1:8B}` and dry-parses it through the C++
ZeRO layout.
Use `scripts/validate_goal_h100.sh source-guards` to check brittle source-level
NCCL contracts such as scalar all-reduce element counts, plus the ZeRO-3
parameter-shard runtime contract, launch-script step contracts, runtime success-marker
contracts, GQA/RoPE source-routing invariants, BF16/Hopper+Blackwell/TK build-contract
invariants, GELU-epilogue source contracts, profile-gate source contracts,
Llama conversion source contracts, rank-0 training-log evidence contracts,
compile-target coverage, and
`goal-complete` coverage before running a distributed job.
Use `scripts/validate_goal_h100.sh gqa-runtime` for the dedicated H100 GQA/RoPE
CUDA comparison against the CPU reference smoke shapes.
Set `NCCL_DIR`, `NCCL_INCLUDE_PATH`, or `NCCL_LIB_PATH` before running the
harness when the target cluster provides NCCL through a module or custom
prefix. The preflight phase first requires an H100/sm90-class GPU by default,
then checks the same NCCL paths the Makefile uses. With
`DEVICE_TEST_TARGET=blackwell`, preflight instead accepts B200/GB200 or
sm_100/sm_103-class datacenter Blackwell devices and skips NCCL/MPI by default.
With `DEVICE_TEST_TARGET=rtx5090`, it accepts RTX 5090/sm_120-class devices and
also skips NCCL/MPI by default. The `cuda-runtime` phase independently checks
the same target contract inside the compiled runtime probe.
`ALLOW_NON_H100=1` bypasses only the GPU class gate for dry compile/debug runs;
it is not valid goal-completion evidence, and `goal-complete` refuses to run
while it is set.
Longer phases such as `profile`, `llama1b-stability`, `gpt2-full`,
`gpt2-two-node`, `llama1b-full`, `llama8b-convert`, and `llama8b-full` must be
requested explicitly because they invoke Nsight Compute, gated conversion,
bounded stability runs, loss-curve comparisons, or full training scripts.
The `goal-complete` phase fail-fast checks `gpt2_124M_bf16.bin`, `ncu` for live
or `.ncu-rep` profile validation, and `sbatch` when the two-node/full 8B phases
are not in validate-only mode. In validate-only mode it checks the pre-existing
two-node reference/candidate logs and 8B checkpoint/log artifacts before
entering `goal-core`.
Use `scripts/validate_goal_h100.sh goal-complete-prereqs` to check those
completion prerequisites without launching `goal-core` or long jobs.
Short H100 gates can replay captured logs instead of launching with
`PREFLIGHT_VALIDATE_ONLY=1`, `CUDA_RUNTIME_VALIDATE_ONLY=1`, `SMOKE_VALIDATE_ONLY=1`,
`GPT2_RUNTIME_VALIDATE_ONLY=1`, `GQA_RUNTIME_VALIDATE_ONLY=1`,
`GPT2_SMOKE_VALIDATE_ONLY=1`, `LLAMA_RESUME_VALIDATE_ONLY=1`, or
`LLAMA1B_STABILITY_VALIDATE_ONLY=1`; point the matching `*_LOG` variables or
output directories at the captured evidence.
`gpt2-two-node` runs `scripts/multi_node/run_gpt2_124M_fs.sbatch` with
`sbatch --wait` and `MAX_STEPS=100` unless `GPT2_TWO_NODE_VALIDATE_ONLY=1` is
set for existing logs.
The training phases do not stop at process exit: `gpt2-smoke`, `zero3-smoke`,
`llama-resume`, `llama1b-stability`, `gpt2-full`, `llama1b-full`, and
`llama8b-full` validate the rank-0 `main.log` after launch.
Set `GPT2_FULL_VALIDATE_ONLY=1` or `LLAMA1B_FULL_VALIDATE_ONLY=1` to validate
existing full-run logs instead of relaunching those single-node full phases.
Set `LLAMA8B_CONVERT_VALIDATE_ONLY=1` to require and validate an existing
`LLAMA8B_CHECKPOINT` instead of attempting the gated HF conversion.
`gpt2-smoke` requires final validation/train metrics and train-loss decrease.
`llama-resume` requires header-validated initial and final checkpoint artifacts
plus final validation/train metrics after the restarted run.
`gpt2-full` checks final validation loss and HellaSwag accuracy at step 18,865;
`goal-complete` requires `GPT2_FULL_EXPECTED_VAL_LOSS`,
`GPT2_FULL_EXPECTED_HELLASWAG`, `GPT2_TWO_NODE_REL_TOL`,
and the smoke/Llama `*_MAX_VAL_LOSS` / `*_MIN_HELLASWAG` thresholds so those
comparisons are explicit. Individual phases still allow some threshold
variables to be omitted for exploratory runs.
The `profile` phase honors `PROFILE_MIN_TENSOR_UTIL=70` by default and fails
the gate if the averaged Nsight Compute tensor-core utilization is lower. It
also runs `PROFILE_GELU_FUSIONS="0 1"` by default, producing separate
`profile_ge0.ncu-rep` and `profile_ge1.ncu-rep` reports for the default GPT-2
MLP path and the opt-in bias+GELU epilogue path.
Set `PROFILE_VALIDATE_ONLY=1 PROFILE_CSV_DIR=/path/to/csv` to validate existing
raw `profile_ge*.csv` exports through the same parser and tensor-util gate
without requiring Nsight Compute on the validation host. Set
`PROFILE_VALIDATE_ONLY=1 PROFILE_REPORT_DIR=/path/to/reports` to validate
existing `profile_ge*.ncu-rep` reports; that mode still needs local `ncu` to
export the raw CSV before parsing.
`profile-parser` checks the same parser and threshold logic against synthetic
raw CSV without requiring `ncu`.

## Dataset preparation

```bash
# GPT-2 (default; uint16 tokens, vocab 50257, magic 20240520 v1)
python dev/data/tinyshakespeare.py
python dev/data/tinystories.py
python dev/data/fineweb.py
python dev/data/hellaswag.py

# Llama-3 (uint32 tokens, vocab 128256, magic 20240801 v7)
python dev/data/tinyshakespeare.py --model_desc llama-3
python dev/data/tinystories.py --model_desc llama-3
python dev/data/fineweb.py        --model_desc llama-3
python dev/data/hellaswag.py      --model_desc llama-3
```

`dataloader.h` dispatches training and eval files on the header magic at load
time; no flag is needed on the C++ side. **Shards are not interchangeable** —
a `gpt-2` shard cannot be loaded by Llama-3 and vice versa.

HellaSwag writes `hellaswag_val.bin` for GPT-2 and `hellaswag_val_llama3.bin`
for Llama-3. The Llama eval file uses uint32 tokens and magic `20240802` v7;
`EvalLoader` consumes both GPT-2 and Llama-3 eval formats.

After preprocessing, run:

```bash
python dev/validate_data_artifacts.py
```

The validator checks exact file sizes, magic/version pairs, token width,
sampled train-token ranges, and the full HellaSwag-style eval example stream.
Use `DATA_ARTIFACT_ARGS="--full-token-scan"` with the H100 harness if you want
the `data-artifacts` phase to scan every training token instead of sampling.
`make test_dataloader && ./test_dataloader` additionally checks the C++
`DataLoader` and `EvalLoader` code paths against synthetic GPT-2 uint16 and
Llama-3 uint32 train/eval files.
