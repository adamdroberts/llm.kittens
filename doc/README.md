# llm.kittens tutorial notes

This directory is the narrative porting archive. It explains how the major
`llm.c` surfaces were mapped onto ThunderKittens, which constraints shaped each
wrapper, and which pieces are still deliberately baseline CUDA while the target
TK kernels are pending.

The operational source of truth is still [`../goal.md`](../goal.md). If this
directory and `goal.md` disagree, update the docs after checking the code.

## Pages

- [`gemm/gemm.md`](gemm/gemm.md) - TK H100 GEMM wrapper, shape constraints,
  bias handling, and the TK-backed backward path.
- [`attention/attention.md`](attention/attention.md) - GPT-style MHA forward
  and backward, padded forward dispatch, fallback status, and the Llama-3
  GQA/RoPE validation gap.
- [`norms/norms.md`](norms/norms.md) - LayerNorm and RMSNorm forward wrappers,
  fused residual paths, supported widths, and backward baselines.
- [`llama3/llama3.md`](llama3/llama3.md) - Llama-3 trainer loop, descriptors,
  checkpoint header, primitive kernels, scripts, and remaining validation work.

## How to read these notes

The pages follow the same split used in the codebase:

1. `train_*.cu` owns model flow and command-line contracts.
2. `llmc/*.cuh` exposes C-style CUDA-callable wrappers.
3. `llmc/tk/*.cuh` owns template-heavy ThunderKittens code.

That split is intentional. It keeps the trainers close to `llm.c`, keeps the
wrappers testable, and prevents `kittens::` details from leaking into every file.

## Current status

The tutorial archive itself is present, but the project is still mid-port.
The largest remaining technical items are:

- H100 validation of TK GQA and the `train_llama3.cu` forward/backward/update loop.
- GPT-2 parity/runtime validation, including the opt-in TK GEMM bias+GELU
  epilogue behind `-ge 1`.
- H100 runtime validation and `ncu` profiling.
