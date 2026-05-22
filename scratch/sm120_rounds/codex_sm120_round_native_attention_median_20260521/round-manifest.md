# SM120 Round Manifest

- run label: `codex_sm120_round_native_attention_median_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_native_attention_median_20260521`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `487`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `7954b6eab412fa2fdffc87838faa368e782c038b9644b2c325815741eeddcf2a` |
| `test_attention` | `True` | `1760032` | `54d9790a06e245b7f914cc6e3f4560b3d8153db60a8c89db7bde058bfdbbc409` |
| `test_layernorm` | `True` | `1237784` | `c279f54142720ada87d2d226d8f7045d5c91b79f2cfbe01c5d70addebc5b3cf7` |
| `test_bias` | `True` | `2048616` | `ef6da470bfb2b8f26028d346f1ba19603d28cfcef4ee11933773ea15bce8c4ae` |
| `test_gelu` | `True` | `1139336` | `d46a46ef5cb112b1843e7b8eab3bf94128192e5fef0acdcc3ca8a5c635291527` |
| `test_fused_classifier` | `True` | `1164032` | `50517fd8bbc6a94b265720d5f3310daa149d11a65878c201932329056fd45f1b` |
| `test_encoder` | `True` | `1165512` | `895f408266158bbe8e41d0af9bf3a58ef6e83bc2aef4775331cc38f961f61058` |
| `test_adamw` | `True` | `1138408` | `8eb1fb91088208895db1fc69a5bed18f06d579513d72a5fd95ec2108f3b25646` |
| `test_global_norm` | `True` | `1138880` | `f9ce81b2f383e0e8cf433436d8ce3d6d60b29887284a24b41b3484ee60e6ea40` |
| `bench_sm120_matmul` | `True` | `2373912` | `ebf7e940b02e68d9252afa0b39a4390a15e1fd07f7f00fd61af5a8b820876e8f` |
| `bench_sm120_attention` | `True` | `1728312` | `731e6672e8408181f58dc92004634dc341d5dedd6c19725676cb39a681c71054` |
| `bench_sm120_layernorm` | `True` | `1233728` | `5f09cc32c7dc8de0aa4f4884da199f121cf4ae28888048f20c6cfef886cd6a2a` |
| `bench_sm120_runtime` | `True` | `2217576` | `ffdaba723c2dfb9fbd1e2df189707e20cdc4c8f62e35a54a08e42b4a193c894e` |
| `train_gpt2cu` | `True` | `3045944` | `832c038c6e454e85555fcc644434bb30bc44e6b72ad0c4bb3a1abf0648523bec` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
