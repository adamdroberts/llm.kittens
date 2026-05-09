# llm.kittens documentation

Operational docs for an in-progress port of [`karpathy/llm.c`](https://github.com/karpathy/llm.c)
onto [`HazyResearch/ThunderKittens`](https://github.com/HazyResearch/ThunderKittens).

The single source of truth for **what is done and what is left** is
[`../goal.md`](../goal.md). Read that first.

## Where to start

| If you are… | Read |
|---|---|
| Skimming the project | [../README.md](../README.md), then [architecture.md](architecture.md) |
| Building / running today | [build-and-run.md](build-and-run.md) |
| Adding a kernel or wrapper | [architecture.md](architecture.md), [kernel-reference.md](kernel-reference.md), [porting-notes.md](porting-notes.md) |
| Picking up the next milestone | [../goal.md](../goal.md), then the milestone page below |
| Wondering why BF16 only | [precision.md](precision.md) |
| Wiring multi-GPU or multi-node | [multi-gpu.md](multi-gpu.md) |
| Llama-3 specifics | [llama3.md](llama3.md) |
| Understanding how the kernels were ported | [../doc/README.md](../doc/README.md) |
| Numerical-parity testing | [testing.md](testing.md) |
| LLM agent landing here | [agents.md](agents.md) |

## Pages

- [architecture.md](architecture.md) — Layered build (`train_*.cu` → `llmc/*.cuh` → `llmc/tk/*.cuh`), allocator alignment, file-by-file responsibilities, dependency Mermaid diagram, and the strict "TK-namespace stays inside `llmc/tk/`" rule.
- [build-and-run.md](build-and-run.md) — Toolchain, `TK_ROOT`, `make` targets, optional flags (`NO_OMP`, `NO_MULTI_GPU`, `NO_USE_MPI`), starter-pack download, data artifact validation, and smoke training command.
- [kernel-reference.md](kernel-reference.md) — Per-kernel mapping table (which kernel is TK-backed vs verbatim from llm.c), shape constraints, head-dim restrictions, current-state vs target-state per kernel.
- [precision.md](precision.md) — Why BF16 is locked in v1, what the FP32 master-weights / accumulators look like, what the v2 story (fp8 / int8 / mxfp8) probably is.
- [multi-gpu.md](multi-gpu.md) — `zero.cuh` covers ZeRO-0/1/2/3 plus NCCL init paths (single-process, MPI, FS, TCP); ZeRO-3 owns local parameter shards and all-gathers into the current full compute layout; H100-pod env vars (`NCCL_NVLS_ENABLE`, `NCCL_IB_HCA`, `NCCL_NET_GDR_LEVEL`).
- [llama3.md](llama3.md) — Llama-3 model spec, descriptor strings, header magic `20240803` v5, RMSNorm + RoPE + GQA + SwiGLU plan, and the remaining GQA/RoPE validation work.
- [testing.md](testing.md) — Test pyramid: host-only data loader smoke plus GEMM/GQA smoke tests → forward-only sanity → numerical parity vs `gpt2_124M_debug_state.bin` → `main.log`-verified 8×H100 reproduction → multi-node sanity → ncu profiling.
- [`../scripts/validate_goal_h100.sh`](../scripts/validate_goal_h100.sh) — executable H100 validation checklist for the remaining `goal.md` runtime gates.
- [porting-notes.md](porting-notes.md) — llm.c → llm.kittens diffs, the rules of the road, allocator alignment, header magic compatibility, gotchas already documented in source comments.
- [../doc/README.md](../doc/README.md) — Narrative tutorial archive for GEMM, attention, normalization, and Llama-3 porting notes.
- [agents.md](agents.md) — Repo-local agent skills index. Tells future LLMs which skill to use for which class of work.

## Convention

Every page is grounded in real files. If a page references a function, header,
or flag, that name exists in this tree (or in the upstream source the comment
points to). If a page contradicts code, code wins — and the page is wrong, not
the code.

The full machine-readable bundle is [`../llms-full.txt`](../llms-full.txt); the
short index for LLMs is [`../llms.txt`](../llms.txt).
