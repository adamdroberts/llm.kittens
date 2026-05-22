# SM120 Optimization Round

- run label: `codex_sm120_round_python_stacks_selection_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_python_stacks_selection_20260520`
- max steps: `3`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- git commit: `0f21747`
- working tree: `475` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
nvidia-smi/NVML metadata query did not return device metadata in this process context

```

## probe_sm120_backend_stacks
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520/round-manifest.md --run-label codex_sm120_round_python_stacks_selection_20260520 --artifact-dir scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520 --train-out-dir log124M/5090_S_codex_sm120_round_python_stacks_selection_20260520 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 0 --run-benchmarks 1 --run-python-stack-benchmarks 1 --run-training 0 --keep-checkpoints 0`


## bench_sm120_matmul
Command: `./bench_sm120_matmul`

