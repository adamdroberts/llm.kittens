# SM120 Round Manifest

- run label: `codex_sm120_round_optional_refresh_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_optional_refresh_20260521`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `497`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `b6a005692fb34395abec1b9aa1f8d3c00e9d43dd77b214fb009a612694a4f1d1` |
| `test_attention` | `True` | `1760032` | `32be4c09b65d55044f22ae9f4746e80dcae586d134465492425c1a7f461fa412` |
| `test_layernorm` | `True` | `1237784` | `8678d8f9d45aea37bc6ad5200ed3d651db85d6fdfb4e3a4b50f6eb536b0d7591` |
| `test_bias` | `True` | `2048616` | `f1417a21bc8470fe26c02f256271fd7f9d7feea55db76607bc5c27dc6c46f3ec` |
| `test_gelu` | `True` | `1139336` | `03bd82907ad6536d43d9c2c2afbf5eb9f348aff0f35155b65d3737c6451a9de7` |
| `test_fused_classifier` | `True` | `1164032` | `7f0a70e78d39b438953c28a66bba3a7c917593454defacfdcaac136fc373ea13` |
| `test_encoder` | `True` | `1165512` | `a60a7f664cea74a21962c6f520e0131058944045ac2c90fb50d4f6eabfcd1c95` |
| `test_adamw` | `True` | `1138408` | `4d254abac439f8306d5f0631758fd13050e4fb50beb8a4be17604d0ec4ea1190` |
| `test_global_norm` | `True` | `1138880` | `74089cfbcaeda767a0524bad7c57d4f2a812776682f6d5ef704157d95d61781a` |
| `bench_sm120_matmul` | `True` | `2373912` | `30b29f14631c9ef3828b42aa889cb4a5c390672ca8df48735b13afd10527f734` |
| `bench_sm120_attention` | `True` | `1728312` | `d550401ab7801be130c62eceab941d2007c235b255fa1d94c776872fe23bbe24` |
| `bench_sm120_layernorm` | `True` | `1233728` | `99bc076eabed6d549840a30167631b7426fce9b887b2de955f20012d43d2f5f9` |
| `bench_sm120_runtime` | `True` | `2217576` | `8e20bf715ccd15bbd2fee040111975527309726207347e364b374e44c67fc659` |
| `train_gpt2cu` | `True` | `3045944` | `b9f5696ba410dd416a312acdb232e5b79a972a94244c7aa51e671126eb216875` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
