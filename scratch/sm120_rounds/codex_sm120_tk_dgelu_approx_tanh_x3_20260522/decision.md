# Candidate Decision

- Candidate: `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` with `LLMK_SM120_APPROX_DGELU_TANH=1`
- Status: correctness rejected
- Build: passed
- First failing gate: `test_matmul`
- Candidate `train_gpt2cu` sha256: `549736f9558be4bee7a6f28fee305f21491c5a3a51ba22b7c6e17a8d9b2ae084`
- Restored selected `train_gpt2cu` sha256: `4374a593bdc123692c38f28c92951abbe914c4f56f96a9fd4e140fa3c4b119da`

## Failure

The candidate passed 9/10 `test_matmul` rows, but failed the GPT-2 fcproj
fused dGELU dInput route:

```text
dInp backward fused dGELU (GPT-2 fcproj route)
max abs diff = 0.5000  (tolerance 0.50)  FAIL
```

No benchmarks or training run were accepted for this candidate because the
correctness gate failed first.

## Restore

The selected stack was rebuilt without the candidate flags:

- cuBLASLt GEMM
- default promoted SM120 attention route
- CUDA-kernel grad-zero
- Torch C++ dresidual-zero
- host scalar AdamW grad scale
- `LLMK_SM120_DPREP_WARPS=3`
- `LLMK_SM120_MEMORY_BLOCK_SIZE=1024`
- `LLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1`
- ZeRO stage 1

After restoration, all nine focused smoke tests passed:
`test_matmul`, `test_attention`, `test_layernorm`, `test_bias`, `test_gelu`,
`test_fused_classifier`, `test_encoder`, `test_adamw`, and `test_global_norm`.
