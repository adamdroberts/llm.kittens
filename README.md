# llm.kittens

GPT-2 / GPT-3 / Llama-3 training in single-binary C++/CUDA, with every GPU
kernel built on top of [ThunderKittens](https://github.com/HazyResearch/ThunderKittens).
Mirrors [karpathy/llm.c](https://github.com/karpathy/llm.c) feature-for-feature
but replaces its bespoke CUDA kernels (and cuBLAS / cuBLASLt / cuDNN) with
TK-backed ones.

> **In-flight port.** The ⬜ checkboxes in [`goal.md`](goal.md) are the canonical
> "what is done / what is left". Today `make all` builds the GEMM smoke test,
> `train_gpt2cu`, and `test_gpt2cu`; `make gpt2_validate` builds the
> forward-only GPT-2 loss gate; `make train_llama3cu` builds the Llama
> trainer loop with TK GQA forward plus supported-shape backward paths;
> `make test_attention`, `make test_layernorm`, and `make test_attention_gqa`
> build the GPT MHA, GPT LayerNorm, and GQA reference smoke harnesses.
> H100 runtime parity and full Llama-3 support are tracked through M2–M8. **Read [`goal.md`](goal.md)
> first.**

## Documentation

| Need | Page |
|---|---|
| Master TODO and milestone status | [`goal.md`](goal.md) |
| Project shape, layering rules, file map | [`docs/architecture.md`](docs/architecture.md) |
| How to build / run today | [`docs/build-and-run.md`](docs/build-and-run.md) |
| Per-kernel mapping (TK vs verbatim from llm.c) | [`docs/kernel-reference.md`](docs/kernel-reference.md) |
| BF16-only invariant and rationale | [`docs/precision.md`](docs/precision.md) |
| ZeRO + multi-node | [`docs/multi-gpu.md`](docs/multi-gpu.md) |
| Llama-3 (M6/M7) plan and new kernels | [`docs/llama3.md`](docs/llama3.md) |
| Test pyramid + tolerances | [`docs/testing.md`](docs/testing.md) |
| H100 goal validation harness | [`scripts/validate_goal_h100.sh`](scripts/validate_goal_h100.sh) |
| Wrapper-PR checklist and common gotchas | [`docs/porting-notes.md`](docs/porting-notes.md) |
| Narrative kernel porting tutorials | [`doc/README.md`](doc/README.md) |
| Repo-local agent skills | [`docs/agents.md`](docs/agents.md), [`.claude/skills/`](.claude/skills/) |
| LLM ingestion artifacts | [`llms.txt`](llms.txt), [`llms-full.txt`](llms-full.txt) |
| Append-only history | [`CHANGELOG.md`](CHANGELOG.md) |

The full docs index is at [`docs/README.md`](docs/README.md).

## Status snapshot

| Milestone | Scope | State |
|---|---|---|
| M1 | Skeleton, Makefile, verbatim ports, BF16 lock, dataset prep, starter pack | ✅ done |
| M2 | Single-GPU GPT-2 forward (TK GEMM + TK MHA-fwd + TK LayerNorm-fwd + `train_gpt2.cu`) | 🟡 GPT-2 forward wrappers, forward-only validation binary, and MHA/LayerNorm smoke harnesses compile-wired; H100 validation pending |
| M3 | Single-GPU GPT-2 backward + `test_gpt2.cu` numerical parity | 🟡 GPT-2 backward kernels and MHA/LayerNorm smoke harnesses compile-wired; H100 parity pending |
| M4 | Multi-GPU NCCL + ZeRO 0/1, full GPT-2 124M reproduction on 8×H100 | 🟡 scripts + log verifier landed; H100 run pending |
| M5 | GPT-3 variants + multi-node + ZeRO 2/3 | 🟡 descriptor + scripts + GPT dry-runs landed; ZeRO-2 compile-wired; ZeRO-3 parameter-shard runtime path compile-wired; H100/NCCL validation pending |
| M6 | Llama-3 1B (RMSNorm, RoPE, GQA, SwiGLU) | 🟡 converter + kernels + trainer loop + dataloaders/checkpoint-resume/logging compile; TK GQA forward + supported-shape bwd compile; TK tile-load RoPE compile-wired for supported shapes; H100 validation pending |
| M7 | Llama-3 8B multi-node | 🟡 Slurm script + converter validation hooks landed; gated HF conversion/runtime validation pending |
| M8 | Polish, profiling, optional bias-fusion | 🟡 docs + profile/log verifier wired; profile helper enforces the 70% tensor-util gate; parser threshold logic has a synthetic CSV check; GEMM epilogue compile/profile-wired; `ncu`/H100 validation pending |

Per-task checkboxes live in [`goal.md`](goal.md).

## Why

`llm.c` is the reference for "small, fast, single-binary GPT training in pure
C/CUDA" — but its kernels are bespoke CUDA + cuBLAS. ThunderKittens is a
tile-oriented kernel framework from the FlashAttention-3 author, H100/Blackwell
first. Putting them together produces an end-to-end open trainer that uses the
most modern Hopper kernel patterns (WGMMA, TMA) and serves as a worked example
for both projects.

## Hardware

- **H100 (sm_90a)** — primary target. WGMMA + TMA + 228 KB shared mem.
- **RTX 5090 (sm_120)** — supported only for the generic device-test path
  (`rtx5090-device`: CUDA runtime probe plus the plain CUDA SwiGLU smoke).
  The TK model kernels remain H100-only.
- **A100, RTX 4090, B200** — not supported in v1. ThunderKittens has
  experimental sm_100/103/120 support; revisit full Blackwell kernels after v1 lands.

The goal harness enforces this at runtime: `scripts/validate_goal_h100.sh
preflight` and `cuda-runtime` target H100/sm90-class devices by default. Set
`DEVICE_TEST_TARGET=rtx5090` for the RTX 5090 device-test probe path, or run
`scripts/validate_goal_h100.sh rtx5090-device`, which forces `DEVICE_ARCH=SM120`.
That path is not valid completion evidence for the unchecked H100 gates in
[`goal.md`](goal.md).

## Precision

BF16 only. ThunderKittens H100 GEMM and MHA kernels are bf16-only and we
inherit the constraint. Master weights, gradient accumulation, and AdamW state
remain FP32 — same invariants as `llm.c`. Compile-time errors out on `ENABLE_FP16`
or `ENABLE_FP32`. Full rationale and the v2 plan: [`docs/precision.md`](docs/precision.md).

## Build (today)

```bash
# Default: assumes ../ThunderKittens is the TK checkout.
make

# Or set TK_ROOT explicitly:
make TK_ROOT=/path/to/ThunderKittens
```

`make` (default) builds `test_matmul`, `train_gpt2cu`, and `test_gpt2cu`.
`gpt2_validate` is the forward-only GPT-2 loss gate; `profile_gpt2cu` is
available as a separate target. `profile_gpt2cu.py --csv-input` exercises the
report parser and tensor-utilization gate from an exported raw CSV, and
`profile_gpt2cu.py --gelu-fusion 1` profiles the opt-in epilogue path. GPT-2's
`-ge 1` flag opts the MLP up-projection
into the TK finish-path bias+GELU epilogue; default remains `-ge 0` until H100
numerics are validated. `train_llama3cu` builds the
Llama trainer loop with TK GQA forward and supported-shape backward where
available, with CUDA fallback elsewhere. TK-supported GQA shapes rotate Q/K
inside the tile-load path; fallback shapes use fused materialization/unpermute.
The Llama trainer also writes model + per-rank optimizer/dataloader checkpoint
state, writes rank-0 `main.log` metrics, and supports `-y 1` resume from the
latest completed checkpoint. H100 runtime validation remains M6 work — see
[`goal.md`](goal.md).

Requirements: CUDA Toolkit ≥ 12.4, GCC ≥ 11 (or Clang ≥ 14), a ThunderKittens
checkout (header-only), optionally NCCL + OpenMPI for multi-GPU / multi-node
(`NCCL_DIR=/path/to/nccl` or an explicit `NCCL_INCLUDE_PATH` /
`NCCL_LIB_PATH` pair is supported for cluster/module installs),
Python ≥ 3.10 for `dev/data/*.py` preprocessing scripts. Full toolchain detail:
[`docs/build-and-run.md`](docs/build-and-run.md).

## Smoke test (today)

```bash
make test_matmul
./test_matmul

make test_attention
./test_attention

make test_layernorm
./test_layernorm

make test_rope
./test_rope

make test_rmsnorm
./test_rmsnorm

make test_swiglu
./test_swiglu

make test_attention_gqa
./test_attention_gqa
python dev/validate_attention_gqa_reference.py
scripts/validate_goal_h100.sh gqa-runtime
python dev/validate_profile_parser.py
python dev/validate_log_tools.py
python dev/validate_llama3_converter.py --cpp-validate --train-binary ./train_llama3cu
python dev/validate_nccl_source.py
python dev/validate_build_contracts.py
python dev/validate_epilogue_source.py
python dev/validate_gqa_source.py
python dev/validate_runtime_markers.py
python dev/validate_training_source.py
python dev/validate_profile_source.py
python dev/validate_llama_conversion_source.py
python dev/validate_goal_harness_coverage.py

make test_dataloader
./test_dataloader

scripts/validate_goal_h100.sh llama-converter-smoke
scripts/validate_goal_h100.sh llama-checkpoint-smoke

make gpt2_validate
./gpt2_validate
```

The H100 harness asserts the final success markers from these runtime binaries,
including `<binary> smoke OK`, `CUDA runtime check passed.`, `gpt2_validate OK`,
and `test_gpt2cu OK`, rather than relying only on process exit status.
Use `scripts/validate_goal_h100.sh host-core` on local machines that can compile
and run host-only checks but do not have a working CUDA runtime.
Use `scripts/validate_goal_h100.sh rtx5090-device` on an RTX 5090 host to build
the generic CUDA probes with `DEVICE_ARCH=SM120` and run the CUDA runtime plus
plain CUDA SwiGLU smoke. `ALLOW_NON_H100=1` may still be used with the
`preflight` and `cuda-runtime` probes only for dry debugging on unsupported GPUs;
real runtime gates still require H100/sm90-class hardware.
`goal-complete` refuses to run with that override set.
On a target machine where every remaining `goal.md` gate should run in one
intentional pass, use
`ALLOW_FULL_GOAL_RUN=1 scripts/validate_goal_h100.sh goal-complete`; this runs
the ZeRO-3 smoke plus long H100/NCCL/profile/conversion/full-run phases after
`goal-core`.
That completion pass requires `GPT2_FULL_EXPECTED_VAL_LOSS` and
`GPT2_FULL_EXPECTED_HELLASWAG` so the GPT-2 full reproduction is checked against
explicit published metrics, plus explicit max/min thresholds for the GPT-2
smoke, GPT-2 two-node comparison, Llama resume, Llama-3 1B stability/full, and
Llama-3 8B full phases. It also fail-fast checks `gpt2_124M_bf16.bin`, `ncu`
for live or `.ncu-rep` profile validation, and `sbatch` when the two-node/full
8B phases are not in validate-only mode; validate-only mode checks the
pre-existing two-node reference/candidate logs and 8B checkpoint/log artifacts
before entering `goal-core`.
Run `scripts/validate_goal_h100.sh goal-complete-prereqs` to check those
completion prerequisites without launching `goal-core` or long jobs.
The short runtime gates can also replay captured evidence: set
`PREFLIGHT_VALIDATE_ONLY=1`, `CUDA_RUNTIME_VALIDATE_ONLY=1`, `SMOKE_VALIDATE_ONLY=1`,
`GPT2_RUNTIME_VALIDATE_ONLY=1`, `GQA_RUNTIME_VALIDATE_ONLY=1`,
`GPT2_SMOKE_VALIDATE_ONLY=1`, `LLAMA_RESUME_VALIDATE_ONLY=1`, or
`LLAMA1B_STABILITY_VALIDATE_ONLY=1` with the matching `*_LOG`/artifact paths.
`GPT2_FULL_VALIDATE_ONLY=1` and `LLAMA1B_FULL_VALIDATE_ONLY=1` validate existing
single-node full-run logs instead of relaunching those phases.
`PROFILE_VALIDATE_ONLY=1 PROFILE_CSV_DIR=...` validates existing raw
`profile_ge*.csv` exports without requiring Nsight Compute on the validation
host. `PROFILE_VALIDATE_ONLY=1 PROFILE_REPORT_DIR=...` validates
`profile_ge*.ncu-rep` reports too, but that path still needs local `ncu` to
export the raw CSV before parsing.
`LLAMA8B_CONVERT_VALIDATE_ONLY=1` requires `LLAMA8B_CHECKPOINT` and validates it
instead of attempting a gated HF conversion.
GPT-2 smoke, Llama resume, and long training phases
validate rank-0 `main.log` with `dev/validate_training_log.py` after launch;
`gpt2-two-node` runs the filesystem Slurm script for 100 steps by default and
compares the single-node vs two-node train-loss curves with
`dev/compare_training_logs.py`. Llama resume and the 8B Slurm gate
also header-validate the expected checkpoint artifacts. The 8B Slurm gate uses
`sbatch --wait` unless `LLAMA8B_FULL_VALIDATE_ONLY=1` is set for an existing
output directory.

Sweeps three forward GEMM shapes (a 1024³ square, the GPT-2 124M MLP
up-projection, and the GPT-2 124M LM head), the opt-in forward bias+GELU
epilogue shape, plus both dWeight `A^T*B` paths: overwrite and accumulated
`+=` through the scratch-backed TK path. All compare against naive bf16
references with FP32 accumulation. Requires an H100 to be
meaningful at runtime.

`test_attention` checks GPT-style packed Q/K/V attention against an independent
CPU reference, including direct TK forward, padded TK forward, fallback
backward, and supported-shape TK backward.

`test_layernorm` checks GPT LayerNorm forward, fused residual+LayerNorm forward,
saved `mean`/`rstd`, and backward `+=` accumulation into `dinp`, `dweight`, and
`dbias` against independent CPU references.

## Quickstart (compile-ready, runtime parity pending)

```bash
# 1. Reference checkpoints + tokenizer for GPT-2 124M.
./dev/download_starter_pack.sh
python dev/validate_gpt2_starter_pack.py --self-test
python dev/validate_gpt2_starter_pack.py

# 2. Tokenize a smoke-test dataset.
python dev/data/tinyshakespeare.py
python dev/validate_data_artifacts.py --self-test
python dev/validate_data_artifacts.py

# 3. Build the trainer.
make train_gpt2cu

# 4. Train 100 steps from the bf16 124M checkpoint.
./train_gpt2cu \
    -i dev/data/tinyshakespeare/tiny_shakespeare_train.bin \
    -j dev/data/tinyshakespeare/tiny_shakespeare_val.bin \
    -e gpt2_124M_bf16.bin \
    -b 4 -t 1024 -x 100 -v 20 -s 0
```

For the full GPT-2 124M reproduction on 8×H100 (M4 milestone), see
`scripts/run_gpt2_124M.sh`. The single-node and multi-node script list is in
[`docs/multi-gpu.md`](docs/multi-gpu.md#per-script-status).
For the bounded Llama-3 1B FineWeb-edu stability gate, use
`scripts/validate_goal_h100.sh llama1b-stability` on the target H100 box.

## Layout

```
llm.kittens/
├── train_gpt2.cu          GPT-2 / GPT-3 entrypoint  [compile-ready; runtime parity pending]
├── train_gpt2.py          PyTorch reference / .bin exporter (verbatim)
├── train_llama3.cu        Llama-3 trainer/checkpoint-resume [compile-ready; H100 validation pending]
├── train_llama3.py        Llama-3 HF→.bin converter
├── test_gpt2.cu           Numerical parity test     [compile-ready; runtime parity pending]
├── profile_gpt2.cu        ncu profiling target      [compile-ready; runtime pending]
├── llmc/                  Kernel + utility headers (C-style API)
│   ├── matmul.cuh         TK-backed GEMM fwd + dInp/dWeight/dbias backward
│   ├── attention.cuh      TK MHA fwd/bwd; padded fwd dispatch; CUDA bwd fallback
│   ├── attention_gqa.cuh  Llama-3 GQA + RoPE attention (TK fwd/supported bwd; CUDA fallback)
│   ├── layernorm.cuh      TK LayerNorm fwd + fused-residual; TK warp-sum bwd reductions
│   ├── rmsnorm.cuh        Llama-3 RMSNorm (TK fwd + fused residual; CUDA bwd)
│   ├── rope.cuh           Llama-3 rotary embeddings (TK; compile-ready)
│   ├── encoder.cuh        token+position embedding (verbatim)
│   ├── gelu.cuh           GELU (verbatim)
│   ├── swiglu.cuh         Llama-3 SwiGLU activation (plain CUDA)
│   ├── fused_classifier.cuh  cross-entropy + softmax (verbatim)
│   ├── adamw.cuh          AdamW step (verbatim)
│   ├── global_norm.cuh    grad-clip norm (verbatim)
│   ├── zero.cuh           NCCL + ZeRO-0/1/2/3; ZeRO-3 parameter shards all-gather into the full compute layout
│   ├── dataloader.h      GPT-2/Llama train+eval loaders (header-dispatched)
│   ├── tokenizer.h, sampler.h, schedulers.h, ...   (verbatim)
│   └── tk/                ThunderKittens template-heavy implementations
│       ├── tk_common.cuh  Bridge: bf16 alias, TK_ALIGN, smem helper, BF16 static_assert
│       ├── gemm_h100.cuh  TK bf16 GEMM in header form (✓)
│       ├── attention_h100.cuh   TK MHA fwd (T%192) + bwd (T%256) H100 kernels
│       ├── attention_gqa_h100.cuh   TK GQA fwd + supported bwd [M6 partial; tile-load RoPE compile; runtime pending]
│       ├── layernorm_tk.cuh     TK LayerNorm forward + fused-residual (✓)
│       ├── rmsnorm_tk.cuh       TK RMSNorm forward + fused-residual wrapper (✓ compile-ready)
│       └── rope_tk.cuh          TK RoPE wrapper (✓ compile-ready)
├── dev/
│   ├── cuda/test_matmul.cu         GEMM smoke test (✓)
│   ├── cuda/test_attention.cu      GPT MHA smoke harness (✓ compile; runtime pending)
│   ├── cuda/test_layernorm.cu      GPT LayerNorm smoke harness (✓ compile; runtime pending)
│   ├── cuda/cuda_runtime_check.cu  CUDA driver/runtime/device probe
│   ├── cuda/test_rope.cu           RoPE smoke harness (✓ compile; runtime pending)
│   ├── cuda/test_rmsnorm.cu        RMSNorm smoke harness (✓ compile; runtime pending)
│   ├── cuda/test_swiglu.cu         SwiGLU smoke harness (✓ compile; runtime pending)
│   ├── cuda/test_attention_gqa.cu  GQA + RoPE smoke harness (✓ compile; runtime pending)
│   ├── data/                  dataset preprocessing (mirror of llm.c/dev/data/)
│   ├── test_dataloader.cpp    host-only GPT-2/Llama DataLoader/EvalLoader smoke
│   ├── download_starter_pack.sh
│   ├── validate_data_artifacts.py  host-only prepared-data artifact validator + self-test
│   ├── validate_attention_gqa_reference.py  CPU-only GQA/RoPE PyTorch reference check
│   ├── validate_profile_parser.py  host-only profile CSV parser/threshold check
│   ├── validate_log_tools.py  host-only training-log expected-metric pass/fail smoke
│   ├── validate_llama3_converter.py  host-only Llama write_model header/payload smoke
│   ├── validate_nccl_source.py  host-only NCCL/ZeRO runtime-contract source guard
│   ├── validate_build_contracts.py  host-only BF16/H100/TK build-contract source guard
│   ├── validate_epilogue_source.py  host-only GPT-2 bias+GELU epilogue/profile source guard
│   ├── validate_gqa_source.py  host-only GQA/RoPE routing source guard
│   ├── validate_runtime_markers.py  host-only runtime success-marker source guard
│   ├── validate_training_source.py  host-only rank-0 training-log evidence source guard
│   ├── validate_profile_source.py  host-only Nsight Compute profile-gate/source-mode guard
│   ├── validate_llama_conversion_source.py  host-only Llama-3.1 8B conversion source guard
│   ├── validate_goal_harness_coverage.py  host-only compile/goal-complete/runtime-evidence coverage guard
│   ├── validate_goal_replay.py  host-only captured-evidence replay smoke
│   └── download_llama3.py     gated HF Llama-3 converter wrapper + synthetic checkpoint validator
├── scripts/                   Single-node + multi-node training scripts  [GPT/GPT-3/Llama scripts ported; runtime pending]
│   └── validate_goal_h100.sh   H100 validation harness for remaining goal gates
├── docs/                      Operational documentation
└── doc/                       In-tree tutorial / porting notes  [present; runtime notes still pending]
```

The rule for `llmc/`: every wrapper in `llmc/*.cuh` exposes a C-style function
signature. Only files under `llmc/tk/` `#include <kittens.cuh>` and use the
`kittens::` namespace. This keeps `train_*.cu` free of template noise. Full
explanation in [`docs/architecture.md`](docs/architecture.md).

## What changed vs. llm.c

| Component | llm.c | llm.kittens |
|---|---|---|
| Matmul | cuBLASLt (with bias + GELU epilogue fusion) | TK H100 bf16 GEMM (`A*B^T` model-weight path); default bias/GELU are separate passes, with opt-in GPT-2 MLP bias+GELU epilogue compile-wired behind `-ge 1` |
| Attention forward | cuDNN flash-attn or fallback CUDA | TK MHA forward (bf16, head_dim ∈ {64, 128}) |
| Attention backward | cuDNN flash-attn-bwd | TK MHA backward |
| LayerNorm forward | hand-written CUDA | TK layernorm (forked, dropout removed, `D` re-templated) |
| LayerNorm backward | hand-written CUDA | hand-written CUDA accumulator with TK warp reductions |
| Encoder, GELU, SwiGLU, fused_classifier, AdamW, global_norm | plain CUDA | plain CUDA, **kept verbatim from llm.c** |
| ZeRO / multi-node init | plain CUDA + NCCL | ZeRO-0/1/2/3 and NCCL init wired; ZeRO-3 parameter shards all-gather into the current full compute layout; H100/NCCL validation pending |
| Precision | BF16 / FP16 / FP32 | BF16 only |
| Allocator alignment | 16 bytes | 128 bytes (TMA descriptors) |
| Build | nvcc + cuBLAS + cuBLASLt + (optional) cuDNN | nvcc + TK headers, single binary |

The principle: **TK earns its keep on tile-MMA ops** (GEMM, MHA, LayerNorm).
Everywhere else, `llm.c`'s plain CUDA kernels are kept verbatim — TK adds
nothing on element-wise / gather-scatter / 1D-reduction workloads.

## License

MIT, matching `llm.c` and ThunderKittens.

## Acknowledgements

- [karpathy/llm.c](https://github.com/karpathy/llm.c) — the structural template
  for this project. Most of `llmc/*` is ported verbatim.
- [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
  — the kernel framework everything tile-shaped is built on.
