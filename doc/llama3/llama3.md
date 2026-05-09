# Porting Llama-3

Llama-3 support is split across model metadata, checkpoint conversion, new
primitive kernels, data preprocessing, and launch scripts. The C++ training
loop is now compile-wired with TK GQA forward plus supported-shape backward and
CUDA fallback elsewhere. Supported TK GQA shapes rotate Q/K inside the tile-load
path, while fallback shapes use fused materialization/unpermute. H100 validation
remains pending.

## Current code map

- C++ trainer loop: [`../../train_llama3.cu`](../../train_llama3.cu)
- Python HF converter: [`../../train_llama3.py`](../../train_llama3.py)
- Download wrapper: [`../../dev/download_llama3.py`](../../dev/download_llama3.py)
- Operational status page: [`../../docs/llama3.md`](../../docs/llama3.md)
- Single-node script: [`../../scripts/run_llama3_1B.sh`](../../scripts/run_llama3_1B.sh)
- Multi-node 8B script:
  [`../../scripts/multi_node/run_llama3_8B_fs.sbatch`](../../scripts/multi_node/run_llama3_8B_fs.sbatch)

## Descriptor surface

`train_llama3.cu` parses these descriptors:

```text
llama3:1B
llama3:8B
llama3.1:8B
```

It also parses bf16 checkpoint headers with magic `20240803` and version `5`.
The parser resolves the model dimensions, computes the parameter layout, and
validates the payload byte count. With the default `-x 0`, the binary runs this
dry path without touching CUDA; with `-x >0`, it enters the Llama
forward/backward/update loop.

The training loop now writes GPT-2-style checkpoints: rank 0 writes the model,
all ranks write optimizer/dataloader state, and `DONE_%08d` gates resume
discovery. `-y 1` resumes from the latest completed checkpoint in the output log
directory. This is compile-checked; restart continuity still needs a real H100
run.

## Parameter layout

The C++ layout is represented by `LlamaParameterTensors` with ten tensors:

```text
wte
ln1w
qkvw
attprojw
ln2w
fcw_up
fcw_gate
fcprojw
lnfw
lm_head
```

The order is deliberate: the checkpoint writer emits Python `c_fc` / Meta `w3`
first as the up projection, then Python `c_fc2` / Meta `w1` as the SwiGLU gate.
The C++ entrypoint validates the checkpoint payload size against this parsed
layout, so truncated or mismatched converter output fails before CUDA startup.

The packed QKV tensor follows Llama GQA shape rather than GPT shape:

```text
(NH + 2 * NKVH) * HS
```

where `NH` is the number of query heads and `NKVH` is the number of KV heads.

## Primitive kernels

The M6 primitive surface currently consists of:

- [`../../llmc/rmsnorm.cuh`](../../llmc/rmsnorm.cuh) and
  [`../../llmc/tk/rmsnorm_tk.cuh`](../../llmc/tk/rmsnorm_tk.cuh) for RMSNorm
  forward and fused residual forward, plus CUDA backward baseline. The
  [`../../dev/cuda/test_rmsnorm.cu`](../../dev/cuda/test_rmsnorm.cu) smoke
  harness covers this path against CPU references.
- [`../../llmc/rope.cuh`](../../llmc/rope.cuh) and
  [`../../llmc/tk/rope_tk.cuh`](../../llmc/tk/rope_tk.cuh) for RoPE forward and
  inverse-rotation backward. The
  [`../../dev/cuda/test_rope.cu`](../../dev/cuda/test_rope.cu) smoke harness
  covers HS=64 and HS=128 against CPU references.
- [`../../llmc/swiglu.cuh`](../../llmc/swiglu.cuh) for plain CUDA SwiGLU
  forward/backward. The
  [`../../dev/cuda/test_swiglu.cu`](../../dev/cuda/test_swiglu.cu) smoke
  harness covers forward, `dgate`, and `dup` against CPU references.
- [`../../llmc/attention_gqa.cuh`](../../llmc/attention_gqa.cuh) for GQA + RoPE
  dispatch: tile-load RoPE plus TK forward/supported backward where available,
  slow CUDA fallback with fused materialization/unpermute elsewhere.
- [`../../llmc/tk/attention_gqa_h100.cuh`](../../llmc/tk/attention_gqa_h100.cuh)
  for the partial TK GQA forward and supported-shape backward kernel paths.

The highest-risk missing piece is H100 numerical validation. The TK path must
keep summing `dK` and `dV` across repeated query heads while rotating Q/K inside
the attention kernel.

## Data and eval files

The dataset scripts under [`../../dev/data/`](../../dev/data/) accept
`--model_desc llama-3`. HellaSwag writes a separate Llama eval file:

```text
hellaswag_val_llama3.bin
```

That file uses uint32 tokens and header magic `20240802` version `7`, because
Llama token IDs do not fit the GPT-2 uint16 eval format. The shared C++
`EvalLoader` can now parse both GPT-2 and Llama-3 eval records by header magic.

## Launch scripts

[`../../scripts/run_llama3_1B.sh`](../../scripts/run_llama3_1B.sh) builds
`train_llama3cu` and launches the intended 1B single-node recipe.

[`../../scripts/multi_node/run_llama3_8B_fs.sbatch`](../../scripts/multi_node/run_llama3_8B_fs.sbatch)
is the 2-node filesystem-rendezvous Slurm script for 8B targeting ZeRO-2. It
is compile-wired through the ZeRO-2 sharded optimizer/reduce-scatter path, but
still waits on real H100/NCCL validation.

Both scripts syntax-check, but both still wait on HF checkpoint availability,
H100 runtime validation, and TK GQA numerical validation.

## Remaining integration path

The expected order is:

1. Run `dev/cuda/test_attention_gqa.cu` on H100 for `B=1, T=128` and `T=256`.
2. Compare the TK GQA forward/backward paths against PyTorch on those smoke
   shapes and establish tolerances.
3. Re-validate the TK GQA tile-load RoPE path against the materialized fallback path.
4. Runtime-validate the compile-wired `train_llama3.cu` forward/backward/update loop on H100.
5. Runtime-validate checkpoint/resume continuity after the first real run.
6. Run the 1B script on H100, then validate the 8B converter and multi-node
   script.
