# SM120 Round Manifest

- run label: `codex_sm120_round_runtime_shape_validator_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_runtime_shape_validator_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_runtime_shape_validator_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `457`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2137216` | `252b9953ddb6c9ba7067b9e1fe85d7cbad8279085530a38856491a4a9fecc3c9` |
| `test_attention` | `True` | `1760032` | `fc42e4d64345d2b4deccec85ce439d442ea6f08015304ad1246c0b975679e4be` |
| `test_layernorm` | `True` | `1237784` | `8ae83f20c4503a2f3501b4e209c10676c0d4bbcc22f95b6a780c84f09044707a` |
| `test_bias` | `True` | `2048616` | `e75b979e664e004b480ace2b3d0d0512742bae56c5841255ff8b6b68c0b5f152` |
| `test_gelu` | `True` | `1139336` | `eb26e498291ee861893894a29c84d36cc37d44d567aab9be8c98056b8af371f0` |
| `test_fused_classifier` | `True` | `1164032` | `d7bbdad65a59ca743b1f8658e6200e2b00a5e8105f66a11be7a1b4f55861931d` |
| `test_encoder` | `True` | `1165512` | `8ea9a0ded2b1081ba1739bf944ed7d706131ce680f3065bb5b04cf4a04bc4372` |
| `test_adamw` | `True` | `1138408` | `189c86f4f48dbe329b76be40bc99c00d1611c67d4ac6bd26255c0d30df8eabb9` |
| `test_global_norm` | `True` | `1138880` | `05b276aac97a6240632e5e0de202581bacd71f61a0df7080e6a319943fc795b6` |
| `bench_sm120_matmul` | `True` | `2373912` | `d432548805ecefa003772692f74894c23d7a424ae520f1bc03e308b6e107fae8` |
| `bench_sm120_attention` | `True` | `1731352` | `d6731585205c882d48d6cccf90a38d29291d4dc29b3c664ad0da1e5fe2f072cf` |
| `bench_sm120_layernorm` | `True` | `1233728` | `4c6d9aa9a2e4b14196ec523542bc32e7e2b7803b556edcb0a5e5e0314a1fdeb0` |
| `bench_sm120_runtime` | `True` | `2199032` | `657b84428e558225858f03b9c5b60bf6b2f6afe0773508f1410bdca2662b195f` |
| `train_gpt2cu` | `True` | `3045944` | `3818434b4c69fed9c8f3ecca1b007a95716487033b198354e7db925950e256c5` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
