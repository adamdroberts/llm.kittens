# SM120 Round Manifest

- run label: `codex_sm120_round_cublas_bwd_default_rerun_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_cublas_bwd_default_rerun_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_cublas_bwd_default_rerun_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `449`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2128288` | `b655b28b1438dfb7b8bae4e44e851fe9fb62458fdc821f5ac221eed95b2963e9` |
| `test_attention` | `True` | `1760032` | `8de0f214ca5ef4dbda3979f1e199f5ebe9ead2ba8ecc4076b8cbed4e83266fd8` |
| `test_layernorm` | `True` | `1237784` | `44057bfca29a2db904eacfa61ff84419360717d566f9814c8ed7ea3ec40085f6` |
| `test_bias` | `True` | `2039504` | `3997caed09a22ac35359542223c1e58c3e2b4a0f7756db61a5e4759f336cdc96` |
| `test_gelu` | `True` | `1139336` | `8d0bf5876e187c77f8133c8becdddd1d441ee984430d02d499e3a006ac3da27c` |
| `test_fused_classifier` | `True` | `1146768` | `ebb2c91bfb7d0c22c469d0995d4543a85c52aad9d2e5ecd475fb6a2ba7f0aab2` |
| `test_encoder` | `True` | `1165512` | `b1a5a1e0e3863e3be826a9d85d1e0f593aefd5cc63163d6fbd08deb0fdeadb9c` |
| `test_adamw` | `True` | `1138408` | `1515f78abb5aead953770f46d5a59b45710545b8bdf871ad6287c3e61604e40b` |
| `test_global_norm` | `True` | `1138816` | `57bdb04b7fc9bb334f06ed4535e770927ccf417f86cfe743517fdc1eb459697f` |
| `bench_sm120_matmul` | `True` | `2344440` | `742cc1b40f0352ae08ae85e419001dce619c047491268374bdab19b4e98c6562` |
| `bench_sm120_attention` | `True` | `1731352` | `4d038f2555d21423da5eb02b2ae043f97661669c59ad336d4cb2e47c72822c85` |
| `bench_sm120_layernorm` | `True` | `1229088` | `7423b9c49249ca9811023e455c833ff52e5fd9c45428b3afd13c5a452fd7cd87` |
| `bench_sm120_runtime` | `True` | `2168552` | `22b11f348e47b21a4b4d53bde06e94964b17b6e2bcaf865e2d8d66a2d50914a1` |
| `train_gpt2cu` | `True` | `3037016` | `7e71371ac2cd1044ffea0d7f649ee7d1c832c0b5afc7338678740e89dfc66460` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
