# SM120 Optimization Round

- run label: `torch_classifier_dry`
- artifact dir: `scratch/sm120_rounds/torch_classifier_dry`
- train output dir: `log124M/5090_S_torch_classifier_dry`
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

## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/torch_classifier_dry/round-manifest.json --markdown-out scratch/sm120_rounds/torch_classifier_dry/round-manifest.md --run-label torch_classifier_dry --artifact-dir scratch/sm120_rounds/torch_classifier_dry --train-out-dir log124M/5090_S_torch_classifier_dry --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 0 --run-correctness 0 --run-benchmarks 0 --run-python-stack-benchmarks 1 --run-training 0 --keep-checkpoints 0`


## bench_sm120_torch_matmul
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_matmul.py --repeats 7 --large-repeats 3`


## bench_sm120_torch_attention
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_attention.py --repeats 7 --warmup 3`


## bench_sm120_torch_classifier
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_classifier.py --repeats 7 --warmup 3`


## bench_sm120_layernorm_python_stacks
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_layernorm.py --rows 65536 --cols 768 3072 --repeats 7 --warmup 3`


## bench_sm120_torch_runtime
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_runtime.py --repeats 7 --warmup 3`


## Correctness Markers

```text
[test_matmul] missing: scratch/sm120_rounds/torch_classifier_dry/test_matmul.log
[test_attention] missing: scratch/sm120_rounds/torch_classifier_dry/test_attention.log
[test_layernorm] missing: scratch/sm120_rounds/torch_classifier_dry/test_layernorm.log
[test_bias] missing: scratch/sm120_rounds/torch_classifier_dry/test_bias.log
[test_gelu] missing: scratch/sm120_rounds/torch_classifier_dry/test_gelu.log
[test_fused_classifier] missing: scratch/sm120_rounds/torch_classifier_dry/test_fused_classifier.log
[test_encoder] missing: scratch/sm120_rounds/torch_classifier_dry/test_encoder.log
[test_adamw] missing: scratch/sm120_rounds/torch_classifier_dry/test_adamw.log
[test_global_norm] missing: scratch/sm120_rounds/torch_classifier_dry/test_global_norm.log
```

## Matmul Benchmarks

```text
missing: scratch/sm120_rounds/torch_classifier_dry/bench_sm120_matmul.log
```

## Attention Benchmarks

```text
missing: scratch/sm120_rounds/torch_classifier_dry/bench_sm120_attention.log
```

## LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/torch_classifier_dry/bench_sm120_layernorm.log
```

## Runtime Benchmarks

```text
missing: scratch/sm120_rounds/torch_classifier_dry/bench_sm120_runtime.log
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/torch_classifier_dry/bench_sm120_torch_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/torch_classifier_dry/bench_sm120_torch_attention.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/torch_classifier_dry/bench_sm120_layernorm_python_stacks.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/torch_classifier_dry/bench_sm120_torch_runtime.log
```

## Training Steps

```text
missing: scratch/sm120_rounds/torch_classifier_dry/train_gpt2cu.log
```
