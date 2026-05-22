# SM120 Optimization Round

- run label: `codex_sm120_dry_libtorch_route_python`
- artifact dir: `scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python`
- train output dir: `log124M/5090_S_codex_sm120_dry_libtorch_route_python`
- max steps: `3`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- LibTorch runtime route: `cxx-api-raw-pointer`
- git commit: `0f21747`
- working tree: `496` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
NVML metadata query did not return device metadata in this process context
NVML shutdown also reported no initialized process-local NVML context

```

## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/round-manifest.md --run-label codex_sm120_dry_libtorch_route_python --artifact-dir scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python --train-out-dir log124M/5090_S_codex_sm120_dry_libtorch_route_python --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 0 --run-correctness 0 --run-benchmarks 0 --run-python-stack-benchmarks 1 --libtorch-runtime-route cxx-api-raw-pointer --run-training 0 --keep-checkpoints 0`


## bench_sm120_torch_matmul
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_matmul.py --repeats 7 --large-repeats 3`


## bench_sm120_cutedsl_matmul
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_cutedsl_matmul.py`


## bench_sm120_triton_matmul
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_matmul.py --repeats 5 --large-repeats 2`


## bench_sm120_torch_attention
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_attention.py --repeats 7 --warmup 3`


## bench_sm120_cudnn_attention
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_cudnn_attention.py --repeats 7 --warmup 3`


## bench_sm120_triton_attention
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_attention.py --repeats 7 --warmup 3`


## bench_sm120_torch_classifier
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_classifier.py --repeats 7 --warmup 3`


## bench_sm120_triton_classifier
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_classifier.py --repeats 7 --warmup 3`


## bench_sm120_layernorm_python_stacks
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_layernorm.py --rows 65536 --cols 768 3072 --repeats 7 --warmup 3`


## bench_sm120_triton_runtime
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_runtime.py --repeats 7 --warmup 3`


## bench_sm120_torch_runtime
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_runtime.py --repeats 7 --warmup 3`


## bench_sm120_libtorch_runtime
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_libtorch_runtime.py --route cxx-api-raw-pointer --repeats 7 --warmup 3`


## Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_matmul.log
```

## Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_attention.log
```

## LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_layernorm.log
```

## Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_runtime.log
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_torch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_torch_runtime.log
```

## LibTorch C++ Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/bench_sm120_libtorch_runtime.log
```

## Training Steps

```text
missing: scratch/sm120_rounds/codex_sm120_dry_libtorch_route_python/train_gpt2cu.log
```
