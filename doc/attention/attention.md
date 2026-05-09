# Porting attention

Attention has two separate tracks in this repo:

- GPT-2/GPT-3 multi-head attention (MHA), where ThunderKittens has H100
  reference kernels for the forward and backward paths.
- Llama-3 grouped-query attention (GQA), where no full matching TK reference
  exists and the repo carries TK supported-shape paths plus slow CUDA fallback.

## GPT MHA source map

- TK source shape: `ThunderKittens/kernels/attention/mha_h100/mha_h100.cu`
- TK wrapper: [`../../llmc/tk/attention_h100.cuh`](../../llmc/tk/attention_h100.cuh)
- Public wrapper: [`../../llmc/attention.cuh`](../../llmc/attention.cuh)
- Trainer caller: [`../../train_gpt2.cu`](../../train_gpt2.cu)

The public wrapper preserves the `llm.c` attention API so the GPT trainer can
stay structurally close to upstream.

[`../../dev/cuda/test_attention.cu`](../../dev/cuda/test_attention.cu) is the
small GPT MHA harness for this path. It compares packed Q/K/V forward and
backward results against an independent CPU causal-attention reference.

## Layout bridge

`llm.c` stores packed QKV as `(B, T, 3, NH, HS)`. The TK MHA kernel wants
separate contiguous tensors in `(B, NH, T, HS)` order. The wrapper therefore
keeps the upstream-style layout glue:

1. Permute packed QKV into separate Q, K, and V buffers.
2. Call the TK attention forward kernel.
3. Unpermute the output back into `(B, T, C)`.

The wrapper uses the `qkvr` storage supplied by the trainer for these layout
buffers. The `att` storage is reused for log-sum-exp scratch.

## Sequence length handling

The TK H100 MHA forward kernel works naturally on sequence lengths aligned to
its tile shape. GPT-2 commonly uses `T = 1024`, which is not divisible by the
fast-path tile multiple used by the imported kernel.

The wrapper handles this by:

- running the direct TK path when the length is already supported;
- padding Q/K/V to a supported `Tpad` otherwise;
- running TK at `Tpad`;
- copying the unpadded result back to the normal output.

This keeps the trainer API unchanged and avoids pushing model-specific padding
logic into `train_gpt2.cu`.

## Backward status

The M3 wrapper ports the TK H100 backward pieces from the same MHA source:

- `bwd_attend_prep_ker`
- `bwd_attend_ker`

The fast path supports `head_dim ∈ {64, 128}` and sequence lengths divisible by
`256` (`T=1024` for GPT-2 is covered). It reads the forward-saved LSE from
`att`, uses the saved forward output as the TK `o` input, writes float Q/K/V
gradients through TK, converts them back to bf16, then reuses the upstream-style
QKV gradient unpermute.

Unsupported backward shapes still fall back to the slow CUDA recompute
baseline. The expected remaining risk is numerical tolerance, not API shape.
The reduction order will differ from cuDNN and from the fallback CUDA
implementation, so [`../../docs/testing.md`](../../docs/testing.md) tracks the
parity/tolerance plan.

The smoke harness currently covers a direct-forward/fallback-backward shape
(`T=192`) and a padded-forward/TK-backward shape (`T=256`). Runtime validation
still needs a compatible H100 driver/runtime.

## Llama-3 GQA status

GQA lives in [`../../llmc/attention_gqa.cuh`](../../llmc/attention_gqa.cuh) with
the partial TK forward and supported-shape backward implementation in
[`../../llmc/tk/attention_gqa_h100.cuh`](../../llmc/tk/attention_gqa_h100.cuh).

The wrapper already does the important semantic work:

- unpack packed Llama Q/K/V from `(B, T, (NH + 2 * NKVH) * HS)`;
- apply forward RoPE to Q and K during materialization;
- map each query head to `kvh = qh / (NH / NKVH)`;
- run TK causal forward where the H100 BF16 shape is supported;
- run TK causal backward where the H100 BF16 shape is supported;
- accumulate `dK` and `dV` across repeated query heads in backward;
- rotate Q/K gradients back through inverse RoPE during packed-gradient unpermute.

The target TK kernel still needs H100 numerical validation. It should keep the
adapted MHA mapping for `n_q_heads`, `n_kv_heads`, and `n_rep`, keep RoPE inside
the supported-shape tile-load path, then validate against a small PyTorch
reference before the full Llama trainer depends on it.
