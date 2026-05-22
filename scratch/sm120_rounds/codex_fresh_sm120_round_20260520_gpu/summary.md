# SM120 Optimization Round

- run label: `codex_fresh_sm120_round_20260520_gpu`
- artifact dir: `scratch/sm120_rounds/codex_fresh_sm120_round_20260520_gpu`
- train output dir: `log124M/5090_S_codex_fresh_sm120_round_20260520_gpu`
- max steps: `3`
- git commit: `0f21747`
- working tree: `430` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Wed May 20 17:07:27 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   29C    P8             56W /  575W |    3277MiB /  32607MiB |      2%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## probe_sm120_backend_stacks
Command: `python3 dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_fresh_sm120_round_20260520_gpu/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_fresh_sm120_round_20260520_gpu/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `python3 dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_fresh_sm120_round_20260520_gpu/round-manifest.json --markdown-out scratch/sm120_rounds/codex_fresh_sm120_round_20260520_gpu/round-manifest.md --run-label codex_fresh_sm120_round_20260520_gpu --artifact-dir scratch/sm120_rounds/codex_fresh_sm120_round_20260520_gpu --train-out-dir log124M/5090_S_codex_fresh_sm120_round_20260520_gpu --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-training 1 --keep-checkpoints 0`


## test_matmul
Command: `./test_matmul`


## test_attention
Command: `./test_attention`


## test_layernorm
Command: `./test_layernorm`

