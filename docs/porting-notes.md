# Porting notes (llm.c → llm.kittens)

The day-to-day notes you need when carrying a kernel or a code path across.
Use this as a checklist before opening a wrapper PR.

The macro plan is in [`../goal.md`](../goal.md). Per-kernel mapping is in
[kernel-reference.md](kernel-reference.md). This page is the operational
companion to those.

## Rules of the road

1. **`train_*.cu` `#include`s only `llmc/*.cuh`.** Never reach into
   `llmc/tk/*.cuh` from training source.
2. **Only `llmc/tk/*` brings `kittens::` into scope.** Everything else uses
   raw `floatX*` and `cudaStream_t`.
3. **`floatX = __nv_bfloat16` is locked.** Do not write paths that branch on
   precision; if you find yourself wanting to, the `static_assert` in
   `tk_common.cuh` will tell you no.
4. **TMA-aligned allocations.** Every parameter / activation buffer that
   passes through a TK kernel must be 128-byte aligned. Use
   `llmk::TK_ALIGN` and `llmk::tk_align()` from
   [`llmc/tk/tk_common.cuh`](../llmc/tk/tk_common.cuh).
5. **One-time `cudaFuncSetAttribute` per TK kernel symbol.** Use the
   `static bool smem_attr_set = false;` latch pattern from
   [`llmc/tk/gemm_h100.cuh`](../llmc/tk/gemm_h100.cuh) (the bottom `launch<>`
   helper). Without it, TK kernels fail to launch on H100.
6. **Element-wise / 1D-reduction / gather-scatter kernels stay verbatim** from
   llm.c. Don't rewrite them on TK; the cost/benefit is wrong.
7. **The bias-grad reduce kernels in `llmc/matmul.cuh` are verbatim from
   llm.c.** Keep them that way; they are the column-reduction kernels for
   `matmul_backward`.

## Diff vs llm.c, point by point

| Area | llm.c | llm.kittens |
|---|---|---|
| Build | nvcc + cuBLAS + cuBLASLt + (optional) cuDNN | nvcc + TK headers, single binary |
| Toolchain | `c++17`, `sm_80` and up | `c++20`, `sm_90a` only (WGMMA + TMA require it) |
| Precision | BF16 / FP16 / FP32 | BF16 only |
| Matmul | cuBLASLt with bias + GELU epilogue fusion | TK GEMM; default bias / GELU are separate passes; GPT-2 MLP up-projection has an opt-in finish-path bias+GELU epilogue behind `-ge 1`; forward uses `A*B^T` to match `(OC, C)` weight files |
| Attention fwd | cuDNN flash-attn (default) or fallback CUDA | TK MHA fwd (head_dim ∈ {64, 128}) |
| Attention bwd | cuDNN flash-attn-bwd | TK MHA bwd |
| LayerNorm fwd | hand-written CUDA | TK layernorm (forked, dropout removed, `D` re-templated, stream param added) |
| LayerNorm bwd | hand-written CUDA | hand-written CUDA accumulator with TK warp reductions |
| Allocator alignment | 16 bytes | 128 bytes (TMA descriptors) |
| Shared-memory opt-in | one cuBLASLt call | `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, ...)` per TK kernel symbol |
| Encoder, GELU, SwiGLU, fused_classifier, AdamW, global_norm | plain CUDA | plain CUDA, **kept verbatim** |
| ZeRO / multi-node init | plain CUDA + NCCL | ZeRO-0/1/2/3 and NCCL init wired; ZeRO-3 parameter shards all-gather into the current full compute layout; H100/NCCL validation pending |
| `cublas_common.h` | full header | empty stub |
| Multi-GPU init | `zero.cuh` (NCCL + MPI/TCP/FS) | `zero.cuh` with llm.c rendezvous paths |

## Wrapper-PR checklist

When porting a kernel, before opening the PR:

- [ ] New file in `llmc/tk/<name>_h100.cuh` that uses `kittens::`.
- [ ] New (or extended) `llmc/<name>.cuh` exposing only C-style functions
      (`floatX*`, `cudaStream_t`).
- [ ] No `<kittens.cuh>` or `kittens::` outside `llmc/tk/`.
- [ ] Source-line comment naming the upstream file (TK kernel, llm.c kernel,
      or both) at the top of every new header.
- [ ] Shape constraints documented and `assert`ed at the wrapper boundary.
- [ ] Smoke test in `dev/cuda/test_<name>.cu` and a Makefile target.
- [ ] [`docs/kernel-reference.md`](kernel-reference.md) row updated (status
      column flipped to ✅ or 🟡).
- [ ] [`../goal.md`](../goal.md) checkbox flipped.
- [ ] [`../CHANGELOG.md`](../CHANGELOG.md) entry appended.
- [ ] Allocator and shared-memory caveats from "Rules of the road" satisfied.

## Common gotchas

### Forgetting the SMEM opt-in

Symptom: TK kernel launch returns `cudaErrorInvalidValue` or hits an SMEM
overflow. Fix: copy the `smem_attr_set` latch from `gemm_h100.cuh::launch<>`.

### Allocator misalignment

Symptom: TMA load reads zeros or garbage; the kernel produces clean numbers
on small shapes and explodes on large ones. Fix: every parameter and
activation buffer needs 128-byte alignment, not 16.

### Mistaking GPT-2 vs Llama-3 shards

Symptom: a shard pattern mixes GPT-2 and Llama-3 files, or the trainer rejects
the data header. Fix: GPT-2 shards are uint16 / vocab 50257 / magic
`20240520` v1; Llama-3 shards are uint32 / vocab 128256 / magic `20240801`
v7. `dataloader.h` dispatches on the header magic and rejects mixed shard
formats. Pass `--model_desc llama-3` to the prep script for Llama-3 runs.

### Head-dim outside {64, 128}

Symptom: TK MHA template fails to instantiate. Fix: HS=64 (GPT-2) and
HS=128 (Llama-3) are the only supported sizes in v1. Anything else needs
a new TK specialization.

### Mixing BF16 and FP16/FP32 paths

Symptom: `static_assert` in `tk_common.cuh` fails, or the `#error` in
`cuda_common.h` fires. Fix: don't. v1 is BF16 only — re-introducing other
precisions is v2 work. See [precision.md](precision.md).

### cuBLASLt epilogue habits

Symptom: looking for the bias / GELU fusion that used to live in
`matmul_forward_cublaslt`. The default v1 path still keeps this explicit: apply
bias via `add_bias_kernel` in [`llmc/matmul.cuh`](../llmc/matmul.cuh), then apply
GELU via [`llmc/gelu.cuh`](../llmc/gelu.cuh). The only compile-wired fused path
today is `matmul_forward_gelu`, used by GPT-2's MLP up-projection when `-ge 1`
is set. Keep it opt-in until H100 numerical validation and `ncu` profiling pass.

## Source pointers (most-cited llm.c lines)

These are the exact upstream sites the wrappers must keep signature-compatible
with or port from verbatim:

- `llm.c/train_gpt2.cu:85-115` — `ParameterTensors`
- `llm.c/train_gpt2.cu:435-517` — `.bin` I/O
- `llm.c/train_gpt2.cu:519-580` — descriptor parser (GPT-2 / GPT-3)
- `llm.c/train_gpt2.cu:646-755` — forward orchestration
- `llm.c/train_gpt2.cu:788-1000` — backward orchestration
- `llm.c/train_gpt2.cu:1365-1415` — CLI flags
- `llm.c/train_gpt2.cu:1712-1900` — training loop
- `llm.c/llmc/matmul.cuh:17,83,231,244` — bias-grad reduce + matmul fwd/bwd signatures
- `llm.c/llmc/attention.cuh:14-83,195,239` — permute/unpermute glue + attention signatures
- `llm.c/llmc/layernorm.cuh:67,233,433,492,467` — LN forward (kernel6, reference), backward (kernel10, port to TK), three signatures
- `llm.c/llmc/fused_classifier.cuh:19,140` — cross-entropy kernel (kept verbatim)
- `llm.c/llmc/zero.cuh` — source for NCCL + ZeRO-0/1 + multi-node init
- `llm.c/train_llama3.py:130-235,245-293,850-920` — Llama-3 architecture spec + checkpoint converter
- `llm.c/dev/data/data_common.py:26-37` — header magics the dataloader matches
- `llm.c/scripts/run_gpt2_124M.sh`, `run_gpt3_125M.sh` — single-node script templates

## ThunderKittens source pointers

- `ThunderKittens/include/kittens.cuh` — TK API entry point
- `ThunderKittens/kernels/gemm/bf16_h100/bf16_h100_gemm.cu` — the GEMM template
  ported into [`llmc/tk/gemm_h100.cuh`](../llmc/tk/gemm_h100.cuh)
- `ThunderKittens/kernels/attention/mha_h100/mha_h100.cu` — `fwd_attend_ker`,
  `bwd_attend_prep_ker`, `bwd_attend_ker` (M2/M3)
- `ThunderKittens/kernels/layernorm/layernorm.cu` — fwd-only LN we fork
  (M2)
- `ThunderKittens/kernels/rotary/rotary.cu` — rotary kernel we wrap for
  Llama-3 (M6)
