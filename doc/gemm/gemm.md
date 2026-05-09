# Porting GEMM

GEMM is the heaviest operator in GPT-2/GPT-3 and the first place where
ThunderKittens must earn its keep. The wrapper keeps the `llm.c` trainer surface
simple while replacing cuBLASLt with a TK H100 bf16 kernel.

## Source map

- TK source shape: `ThunderKittens/kernels/gemm/bf16_h100/bf16_h100_gemm.cu`
- TK wrapper: [`../../llmc/tk/gemm_h100.cuh`](../../llmc/tk/gemm_h100.cuh)
- Public wrapper: [`../../llmc/matmul.cuh`](../../llmc/matmul.cuh)
- Trainer caller: [`../../train_gpt2.cu`](../../train_gpt2.cu)

`llmc/tk/gemm_h100.cuh` exposes H100 specializations in header form:

- `matmul_default<2,4,8>` for normal `A*B` projections where `N % 256 == 0`.
- `matmul_small_n<2,2,8>` for smaller-N `A*B` cases where `N % 128 == 0`.
- `_nt` variants of both aliases for `A*B^T`, used by `matmul_forward` because
  llm.c stores dense model weights as `(OC, C)`.
- `_tn` variants of both aliases for `A^T*B`, used by dWeight backward.
- `_nt_bias_gelu` variants for the opt-in GPT-2 MLP up-projection path. They
  add bias, store the pre-GELU auxiliary buffer, and write GELU output from the
  TK finish path.

Both paths use the same Hopper pattern from the TK reference: persistent grid,
TMA producer, WGMMA consumer, and bf16 inputs with fp32 accumulation inside the
tile pipeline.

## Public API

The trainer calls a C-style wrapper:

```cpp
matmul_forward(out, inp, weight, bias, B, T, C, OC, stream);
```

The logical matrix multiply is:

- `M = B * T`
- `K = C`
- `N = OC`
- `out[M, N] = inp[M, K] * weight[N, K]^T`

Current constraints are asserted in [`../../llmc/matmul.cuh`](../../llmc/matmul.cuh):

- `M % 128 == 0`
- `K % 64 == 0`
- `N % 128 == 0`

These match the current TK H100 tile assumptions and the model families in
scope for v1.

## Bias and GELU

`llm.c` used cuBLASLt epilogues for bias and sometimes fused GELU. The default
v1 TK wrapper intentionally splits that work:

1. Run TK GEMM.
2. Apply bias with `add_bias_kernel` when `bias != nullptr`.
3. Run GELU as the existing plain CUDA `gelu_forward` when the model needs it.

GPT-2's MLP up-projection can opt into `matmul_forward_gelu`, which uses a TK
finish-path epilogue to add bias, write the pre-GELU buffer needed by backward,
and write the GELU activation. `train_gpt2cu -ge 1` selects this path when the
shape is supported; default remains `-ge 0` until H100 numerical validation and
`ncu` profiling are done.

## Backward status

The current `matmul_backward` in [`../../llmc/matmul.cuh`](../../llmc/matmul.cuh)
is mixed:

- `dinp = dout * weight` uses the TK `A*B` path because `weight` is already
  row-major `(OC, C)`.
- `dweight = dout^T * inp` uses TK `A^T*B` when the destination is known to be
  zero.
- Accumulated `dweight += dout^T * inp` uses TK `A^T*B` into the trainer's
  aligned `matmul_scratch` buffer, then adds that temporary into the gradient
  tensor.
- `dbias = column_sum(dout)` uses the verbatim llm.c reduction kernels when
  the auxiliary buffer is available.

The slow dWeight CUDA baseline remains only as a fallback for unsupported shapes
or missing scratch capacity.

## Validation

[`../../dev/cuda/test_matmul.cu`](../../dev/cuda/test_matmul.cu) is the current
GEMM smoke test. It compares TK output against naive bf16 references with fp32
accumulation for:

- 1024 x 1024 x 1024 square GEMM.
- GPT-2 124M MLP up-projection.
- GPT-2 124M LM head.
- Opt-in forward bias+GELU epilogue for a supported default-kernel shape.
- Non-accumulating dWeight `A^T*B`.
- Accumulated dWeight `A^T*B` plus add.

The smoke test requires an H100 to be meaningful at runtime. Compile coverage is
included in the default `make all` path.
