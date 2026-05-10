---
name: llm-kittens-port
description: Carry the in-flight port of llm.c onto ThunderKittens forward. Use whenever the user asks to implement the next milestone, port a kernel, add a TK wrapper, or modify llmc/* or train_*.cu in this repo.
---

# llm-kittens-port

Use this skill when working inside `llm.kittens/` and the request is to
**advance the port** — implement a milestone, port a kernel from llm.c or
ThunderKittens, add a wrapper, or extend the trainer source. Do not use it
for unrelated repository work.

## Step 0 — read the plan, in this order

1. [`goal.md`](../../../goal.md) — the canonical TODO. Find the milestone
   the user is asking about. Note which boxes are checked vs unchecked.
2. [`docs/architecture.md`](../../../docs/architecture.md) — confirm the
   layering rule before touching any file.
3. [`docs/kernel-reference.md`](../../../docs/kernel-reference.md) — find
   the row for the kernel in question; note shape constraints, head-dim
   limits, and which llm.c / TK source the wrapper is meant to mirror.
4. [`docs/porting-notes.md`](../../../docs/porting-notes.md) — the
   wrapper-PR checklist and common-gotchas list.

## Hard rules (non-negotiable)

- `train_*.cu` `#include`s only `llmc/*.cuh`. Never `llmc/tk/*.cuh`.
- Only files under `llmc/tk/` may bring `kittens::` into scope or
  `#include <kittens.cuh>` / `<prototype.cuh>`.
- `llmc/*.cuh` wrappers expose C-style signatures (`floatX*`, `cudaStream_t`).
- `floatX = __nv_bfloat16` is locked. Don't write paths that branch on
  precision — the `static_assert` in `llmc/tk/tk_common.cuh` will fail.
- All TK-bound buffers must be 128-byte aligned. Use `llmk::TK_ALIGN` and
  `llmk::tk_align()`.
- TK kernel launches must opt into the full H100 dynamic SMEM budget. Copy
  the `static bool smem_attr_set = false;` latch from
  `llmc/tk/gemm_h100.cuh::launch<>`.
- Element-wise / 1D-reduction / gather-scatter kernels stay verbatim from
  llm.c. Don't TK-ify them.

## How to add a TK-backed kernel wrapper

1. Make a file in `llmc/tk/<name>_h100.cuh`. It uses templates and
   `kittens::`. Comment header names the upstream TK source (and llm.c
   source, if applicable). Pattern: see [`llmc/tk/gemm_h100.cuh`](../../../llmc/tk/gemm_h100.cuh).
2. Make a file in `llmc/<name>.cuh`. It exposes plain C-style functions
   over `floatX*` and `cudaStream_t`. It `#include`s `tk/<name>_h100.cuh`
   but **no** `<kittens.cuh>` directly. Shape constraints are `assert`ed
   at the wrapper boundary. Pattern: see [`llmc/matmul.cuh`](../../../llmc/matmul.cuh).
3. Add a smoke test at `dev/cuda/test_<name>.cu` that compares against a
   naive bf16 reference with FP32 accumulation. Pattern: see
   [`dev/cuda/test_matmul.cu`](../../../dev/cuda/test_matmul.cu).
4. Wire the test target into the [Makefile](../../../Makefile) (look at the
   `test_matmul` rule).
5. Update the row in [`docs/kernel-reference.md`](../../../docs/kernel-reference.md)
   (status flag).
6. Update the checkbox in [`goal.md`](../../../goal.md).
7. Append a [`CHANGELOG.md`](../../../CHANGELOG.md) entry (what + why,
   verification performed).

## How to add a "verbatim from llm.c" port

1. Copy the file from `../llm.c/llmc/<name>.{cuh,h}` byte-for-byte.
2. Adjust `#include` paths if needed; keep behaviour identical.
3. Comment at the top: "ported verbatim from llm.c/llmc/<name>.{cuh,h} as of
   <commit>". Source-line references in existing comments stay.
4. No tests required — verbatim ports inherit upstream's testing.
5. Update [`docs/kernel-reference.md`](../../../docs/kernel-reference.md) and
   [`goal.md`](../../../goal.md). Append a [`CHANGELOG.md`](../../../CHANGELOG.md)
   entry.

## When *not* to use this skill

- Pure documentation refactors. Use the `deep-documentation` skill instead.
- Debugging a numerical-parity failure. Read [`docs/testing.md`](../../../docs/testing.md)
  first — the tolerance table and the test pyramid will narrow the bug
  faster than re-reading source.
- Anything that would relax `floatX = __nv_bfloat16` or `sm_90a` constraints.
  Those are v2 work, not v1.

## Verification before reporting "done"

- The Makefile target compiles cleanly (`make test_matmul` or the new
  per-kernel test).
- The smoke test passes on H100 (note explicitly if you couldn't run it —
  e.g. no H100 available locally).
- All three: [`goal.md`](../../../goal.md),
  [`docs/kernel-reference.md`](../../../docs/kernel-reference.md), and
  [`CHANGELOG.md`](../../../CHANGELOG.md) reflect the change.

## Pointers to upstream sources

The full list lives in [`docs/porting-notes.md`](../../../docs/porting-notes.md#source-pointers-most-cited-llmc-lines).
The most commonly referenced ones:

- `../llm.c/train_gpt2.cu:85-115,519-580,646-755,788-1000,1365-1415,1712-1900`
- `../llm.c/llmc/{matmul,attention,layernorm,fused_classifier,zero}.cuh`
- `../llm.c/train_llama3.py:130-235,245-293,850-920`
- `../ThunderKittens/include/kittens.cuh`
- `../ThunderKittens/kernels/gemm/bf16_h100/bf16_h100_gemm.cu`
- `../ThunderKittens/kernels/attention/mha_h100/mha_h100.cu`
- `../ThunderKittens/kernels/layernorm/layernorm.cu`
- `../ThunderKittens/kernels/rotary/rotary.cu`
