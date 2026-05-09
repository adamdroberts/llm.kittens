# Porting normalization kernels

Normalization is split between GPT-style LayerNorm and Llama-style RMSNorm.
Both expose plain C-style wrappers under `llmc/` and keep ThunderKittens details
under `llmc/tk/`.

## LayerNorm

Source map:

- TK wrapper: [`../../llmc/tk/layernorm_tk.cuh`](../../llmc/tk/layernorm_tk.cuh)
- Public wrapper: [`../../llmc/layernorm.cuh`](../../llmc/layernorm.cuh)
- Smoke harness: [`../../dev/cuda/test_layernorm.cu`](../../dev/cuda/test_layernorm.cu)
- GPT caller: [`../../train_gpt2.cu`](../../train_gpt2.cu)

The TK fork is based on the ThunderKittens layernorm example, but adjusted for
this trainer:

- dropout removed;
- hidden width re-templated over the model widths used here;
- internal device synchronization removed;
- `cudaStream_t` launchers added;
- `mean` and `rstd` saved because GPT backward needs them.

Supported widths are:

```text
768, 1024, 1280, 1600, 2048, 4096
```

The public functions are:

```cpp
layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C, stream);
fused_residual_forward5(residual, normed, mean, rstd,
                        inp1, inp2, weight, bias, N, C, stream);
```

For supported widths, these dispatch to TK. For unsupported widths, the wrapper
falls back to the CUDA baseline in the same header.

## LayerNorm backward status

Backward keeps the llm.c accumulator structure because it is already a
hand-written kernel that preserves gradient-accumulation semantics. The M3 port
replaces the row-wise warp reductions with a TK `kittens::warp::sum` helper
while preserving the rest of the partial-sum path:

- compute row-wise sums for `dinp`;
- accumulate `dweight` and `dbias`;
- keep the existing cross-block accumulation pattern.

[`../../dev/cuda/test_layernorm.cu`](../../dev/cuda/test_layernorm.cu) adds
CPU-reference coverage for forward, fused residual+LayerNorm forward,
`mean`/`rstd`, and backward accumulation into `dinp`, `dweight`, and `dbias`.
This path is compile-wired; H100 runtime parity is still pending.

## RMSNorm

Source map:

- TK wrapper: [`../../llmc/tk/rmsnorm_tk.cuh`](../../llmc/tk/rmsnorm_tk.cuh)
- Public wrapper: [`../../llmc/rmsnorm.cuh`](../../llmc/rmsnorm.cuh)
- Llama surface: [`../../train_llama3.cu`](../../train_llama3.cu)

RMSNorm mirrors the LayerNorm structure but removes mean subtraction and bias.
The forward equation is:

```text
out = inp * rsqrt(mean(inp^2) + eps) * weight
```

The fused residual path first writes:

```text
residual = inp1 + inp2
```

then normalizes `residual` with the RMSNorm equation.

Supported widths match LayerNorm:

```text
768, 1024, 1280, 1600, 2048, 4096
```

## RMSNorm backward status

[`../../llmc/rmsnorm.cuh`](../../llmc/rmsnorm.cuh) includes a plain CUDA
backward baseline for `dinp` and `dweight`. That is enough for compile coverage
and future Llama trainer integration. The
[`../../dev/cuda/test_rmsnorm.cu`](../../dev/cuda/test_rmsnorm.cu) smoke target
adds CPU-reference coverage for forward, fused-residual forward, `rstd`,
`dinp`, and `dweight`, but it has not been runtime-validated on the full model
path.

Unlike LayerNorm, RMSNorm is part of M6 Llama work, so its final performance
priority depends on runtime validation of TK GQA and the full
`train_llama3.cu` loop.
