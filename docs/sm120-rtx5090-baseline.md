# SM120 RTX 5090 Baseline

This baseline is the fastest validated SM120 setup from the pre-per-round
commit phase that the user identified as the keeper baseline. It is the
reference point for future "best of the best" kernel selection work, regardless
of whether the winning path is ThunderKittens, cuBLASLt, Triton, or a mixed
selector.

## Source State

- Historical record commit: `2255fcd4`
- Restored code state: `2255fcd4` source scope plus the later SM120
  no-master default from `d818239`
- Restored source scope:
  - `Makefile`
  - `train_gpt2.cu`
  - `llmc/attention.cuh`
  - `llmc/matmul.cuh`
  - `llmc/tk/attention_sm120.cuh`
  - `llmc/tk/gemm_sm120.cuh`
  - `dev/cuda/bench_sm120_matmul.cu`
  - `dev/cuda/test_matmul.cu`
  - `docs/cli-reference.md`
  - `docs/kernel-reference.md`

## Build

Use the default SM120 cuBLASLt-backed GEMM path:

```bash
make -j test_matmul test_attention bench_sm120_matmul train_gpt2cu \
    DEVICE_ARCH=SM120 \
    SM120_USE_CUBLASLT_GEMM=1 \
    NO_MULTI_GPU=1 \
    NO_USE_MPI=1
```

The baseline Makefile defines `LLMK_SM120_USE_CUBLASLT_GEMM` for this build
and links `-lcublasLt -lcublas`. It does not define
`LLMK_SM120_CACHE_CUBLASLT_PLANS` by default in this restored baseline.

## Relevant Defaults

- `FORCE_NVCC_O=3`
- `SM120_USE_CUBLASLT_GEMM=1`
- SM120 trainer builds disable FP32 master weights by default
  (`use_master_weights=0`; pass `-w 1` to force the slower master-copy path)
- SM120 cuBLASLt trainer builds keep fused GELU by default (`gelu_fusion=1`)
- Pure SM120 TK trainer builds default to explicit GELU (`gelu_fusion=0`)
- SM120 packed-QKV attention fast path is enabled
- `LLMK_SM120_ATTN_FWD_BLOCK=32`
- `LLMK_SM120_ATTN_BWD_BLOCK=16`
- `LLMK_SM120_BIAS_BLOCK_SIZE=512`
- `LLMK_SM120_K_TILE=32`
- `LLMK_SM120_SUPER_M=8`
- `LLMK_SM120_HUGE_N_K_TILE=64`
- `LLMK_SM120_GRAD_K_TILE=64`
- `LLMK_SM120_DINP_SUPER_M=LLMK_SM120_SUPER_M`
- `LLMK_SM120_DWEIGHT_SUPER_M=2`
- `LLMK_SM120_INPLACE_LAYOUT_SWAP=1`
- `LLMK_SM120_FAST_DGELU=1`
- `LLMK_SM120_APPROX_DGELU_TANH=1`

## Validation Command

Use the TinyStories command capped at 3 training steps. The SM120 source default
must print `use_master_weights disabled` and `gelu_fusion 1`.

```bash
./train_gpt2cu \
    -i "dev/data/tinystories/TinyStories_train.bin" \
    -j "dev/data/tinystories/TinyStories_val.bin" \
    -o "log124M/5090_S" \
    -v 250 -s 20000 -g 144 \
    -h 0 \
    -b 64 -t 1024 -d 524288 \
    -r 0 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 -q 0.0 -u 700 -n 5000 \
    -y 0 \
    -e "d12" \
    -x 3
```

## Recorded Result

Final restored-default cleanup validation on RTX 5090:

- `test_matmul`: passed `8/8`
- `test_attention`: passed all 3 smoke shapes, including packed-QKV
- TinyStories 3-step average: `2525.99 ms`
- Step timings: `2529.42 ms`, `2529.39 ms`, `2522.60 ms`
- Throughput range: `207.3k` to `207.6k tok/s`

The supplied llm.c baseline averaged `2818.23 ms` over 3 steps, so this SM120
cuBLASLt-backed baseline is about `10.4%` faster.

Refreshed verification after restoring the code state on 2026-05-19:

- `test_matmul`: passed `8/8`
- `test_attention`: passed all 3 smoke shapes, including packed-QKV
- TinyStories 3-step average: `2508.27 ms`
- Step timings: `2510.30 ms`, `2506.36 ms`, `2510.18 ms`
- Throughput range: `208.9k` to `209.2k tok/s`
- Printed runtime knobs: `use_master_weights disabled`, `gelu_fusion 1`

## Kernel Gap

This baseline is not a pure-TK win. The final `bench_sm120_matmul` pass still
showed direct pure-TK GEMM behind cuBLASLt on every GPT-2 row, with the worst
ratio at attention-projection dWeight: `1.77x` slower than cuBLASLt.
The refreshed 2026-05-19 benchmark still showed material pure-TK dInput and
dWeight rows behind cuBLASLt, with attention-projection dWeight worst at
`1.74x`. A few forward rows were competitive or faster in that refresh, so the
remaining baseline gap is concentrated in the material backward GEMM rows.
