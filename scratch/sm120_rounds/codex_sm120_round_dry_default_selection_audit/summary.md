# SM120 Optimization Round

- run label: `codex_sm120_round_dry_default_selection_audit`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit`
- train output dir: `log124M/5090_S_codex_sm120_round_dry_default_selection_audit`
- max steps: `3`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- git commit: `0f21747`
- working tree: `488` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
NVML probe did not initialize in this dry-run process context; this is not GPU availability evidence.
NVML shutdown also reported no initialized process-local NVML context.

```

## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/round-manifest.md --run-label codex_sm120_round_dry_default_selection_audit --artifact-dir scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit --train-out-dir log124M/5090_S_codex_sm120_round_dry_default_selection_audit --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 0 --run-correctness 0 --run-benchmarks 0 --run-python-stack-benchmarks 0 --run-training 0 --keep-checkpoints 0`


## Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_matmul.log
```

## Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_attention.log
```

## LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_layernorm.log
```

## Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_runtime.log
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_torch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/bench_sm120_torch_runtime.log
```

## Training Steps

```text
missing: scratch/sm120_rounds/codex_sm120_round_dry_default_selection_audit/train_gpt2cu.log
```

## write_sm120_current_selection
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_current_selection.py --json-out scratch/sm120_rounds/current-sm120-selection.json --markdown-out scratch/sm120_rounds/current-sm120-selection.md`


## audit_sm120_optimization_goal
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/audit_sm120_optimization_goal.py --selection-json scratch/sm120_rounds/current-sm120-selection.json --selection-md scratch/sm120_rounds/current-sm120-selection.md --json-out scratch/sm120_rounds/current-sm120-audit.json --markdown-out scratch/sm120_rounds/current-sm120-audit.md`
