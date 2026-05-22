# SM120 Round Manifest

- run label: `codex_sm120_round_python_stacks_selection_20260520_escalated`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_escalated`
- train output dir: `log124M/5090_S_codex_sm120_round_python_stacks_selection_20260520_escalated`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `475`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2137216` | `94bdbbbd80373f69cf6afd1018a870e3d3f556bc471161f7b5c1fd32f6274270` |
| `test_attention` | `True` | `1760032` | `ff6a927ff88cb2ace9ddb07daf63cba1ee54336b865acd57d090763d980f0c40` |
| `test_layernorm` | `True` | `1237784` | `50dd262812cfd9f6ba305ab5b57b29f3257ff4c020acfdcb4fe3db70d55db20d` |
| `test_bias` | `True` | `2048616` | `2c6ce789a62f077a1e8bab709dae2925f985e1e49797b22d1ee7e497a660fe7d` |
| `test_gelu` | `True` | `1139336` | `94969179fc3fabef2b24d30b56514e7d0fcfb779be5db45314c17dcbf017dcc0` |
| `test_fused_classifier` | `True` | `1164032` | `3ddcfe66f126a798225d86008641a869453291176f85461f6fb75c63b8069ad5` |
| `test_encoder` | `True` | `1165512` | `8fe4e22bfe7e2fac997fc8d7abbd4890fbae0b30208e6f256c9fb4122a14217c` |
| `test_adamw` | `True` | `1138408` | `0b00c7f3ff0dd84d375e64ade254a2a3c4fc926677895e52efce2336892ea6ca` |
| `test_global_norm` | `True` | `1138880` | `6eb864e489d3b6fcf330822e03a8f9d54b404410d3bd1e01408bda3ae2d6e907` |
| `bench_sm120_matmul` | `True` | `2373912` | `e887ed5f6f15cfbcb7642c3188b416377c3126d802e10a903a851637cc9ef391` |
| `bench_sm120_attention` | `True` | `1731352` | `1d1ae364540b3c36507a1b94fbed24ec0388ea378cd9f92518944068d4e55371` |
| `bench_sm120_layernorm` | `True` | `1233728` | `76bd782b7402df4a4399f20a06716251d398f1f93128a410efdff30eb4a9b554` |
| `bench_sm120_runtime` | `True` | `2199032` | `04a1eb89a9b004add50a81aa575f33219fb28bfb0770fcc5777199cac7cf41c3` |
| `train_gpt2cu` | `True` | `3045944` | `8bf90f6d3e12caa8fe06471ee759c5bde897f0aa47e829784bff976e1f7ccfaa` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
