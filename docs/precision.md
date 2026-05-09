# Precision

llm.kittens v1 is **BF16 only** for activations and parameters on the GPU.
Master weights, gradient accumulation, and AdamW state remain FP32 — same
invariants as `llm.c`.

This is the only feature-parity deviation from `llm.c`.

## Why

ThunderKittens H100 GEMM and MHA kernels are bf16. Forcing FP16/FP32 would
require new TK kernel specializations. The cost of writing those is high; the
benefit is low (BF16 is the consensus mixed-precision dtype for training in
2024+). v2 will more likely adopt FP8 / int8 / mxfp8 (TK already ships H100
kernels for those) than re-introduce FP16/FP32.

## What is locked, where

- [`llmc/cuda_common.h`](../llmc/cuda_common.h):
  ```cpp
  #if defined(ENABLE_FP32) || defined(ENABLE_FP16)
  #error "llm.kittens v1 is BF16-only..."
  #endif
  typedef __nv_bfloat16 floatX;
  #define PRECISION_MODE PRECISION_BF16
  ```
  Compile-time `#error` if anyone passes `-DENABLE_FP16` or `-DENABLE_FP32`.

- [`llmc/tk/tk_common.cuh`](../llmc/tk/tk_common.cuh):
  ```cpp
  static_assert(std::is_same_v<floatX, __nv_bfloat16>,
                "llm.kittens v1 requires PRECISION=BF16. ThunderKittens H100 "
                "GEMM/MHA kernels are bf16; FP16/FP32 paths are not implemented.");
  ```
  Hard `static_assert` at the bridge layer, so the GEMM template fails to
  instantiate on a non-BF16 `floatX`.

- The Makefile sets `-DENABLE_BF16` and never sets `-DENABLE_FP16` /
  `-DENABLE_FP32`. The `PrecisionMode` enum exists in `cuda_common.h` for
  symbol-name parity with llm.c, but `PRECISION_MODE` is hard-coded to
  `PRECISION_BF16`.

## What stays FP32

Following llm.c's invariants:

- **Master weights** — kept FP32, downcast to BF16 only for the forward path.
- **Gradient accumulation** — when grad-accum > 1, partial gradient sums use
  FP32 buffers; the BF16 grad is written at the end.
- **AdamW state** (`m`, `v`) — FP32 always.
- **`global_norm` reduction** — FP32 accumulators.
- **Stochastic rounding** — FP32 → BF16 rounding uses the Philox-based
  `stochastic_rounding` helper from `llmc/cuda_utils.cuh`. Verbatim from llm.c.
- **`mfu.h` arithmetic** — FP32, host-side.

## Practical numerical implications

- **Loss and gradient noise.** BF16 has 8-bit mantissa (~3e-3 relative). For
  the GPT-2 family this is well within the noise floor. We expect loss curves
  indistinguishable from llm.c's BF16 path.
- **Test tolerances will widen.** TK MHA bwd uses a different reduction order
  than cuDNN flash-attn, so per-tensor mismatches against the PyTorch reference
  will be ~1e-2 not ~1e-3. We maintain a tolerance table in
  [testing.md](testing.md) and expect ~10× looser bounds on attention bwd
  tensors than llm.c uses.
- **Numerical parity vs llm.c BF16 path.** Should match within bf16 noise on
  forward + AdamW step, except for attention bwd (different reduction order).

## What v2 might look like

Two plausible directions; pick one:

1. **FP8 / mxfp8 training.** TK ships H100 kernels for these. The big-ticket
   change is master-weight quantization (E4M3 vs E5M2 vs mxfp8 scaling), the
   matmul wrapper grows a `dtype` template parameter, and AdamW gains an FP8
   variant. Aligns with the TK ecosystem direction.
2. **Reinstate FP16 / FP32.** Useful for parity with the rest of llm.c, but
   needs new TK template specializations. Lower priority unless a specific
   user request lands.

Either way, removing the `static_assert` in `tk_common.cuh` and the `#error`
in `cuda_common.h` is the first PR.
