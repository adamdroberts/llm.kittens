# SM120 Round Manifest

- run label: `codex_sm120_round_adamw_nomaster_specialized_x10_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_adamw_nomaster_specialized_x10_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_adamw_nomaster_specialized_x10_20260520`
- device arch: `SM120`
- max steps: `10`
- git commit: `0f21747`
- changed paths: `467`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2137216` | `3a4fed96ee6b88b1caf6f025df7b510e07bd13932a29fc14b2394d82fda82d30` |
| `test_attention` | `True` | `1760032` | `a2fe57fc94fc034de18d59415af46e1224d016dd08fc8c45bcd67f1954253700` |
| `test_layernorm` | `True` | `1237784` | `9bba12dc0fa33beb7ab024dc964a0a4dc37a711a605d693535da36c92a287130` |
| `test_bias` | `True` | `2048616` | `78db6d366c536ef97ee5967615edfac766570500cd1bd6b401d3502b18802813` |
| `test_gelu` | `True` | `1139336` | `06148080ac9f6df0bc5bab8365c168e628fd963fbb71030da1f6c9196bc790f6` |
| `test_fused_classifier` | `True` | `1164032` | `3fb1ef9732dc824b676fc52611a55396d059d2b014f904104c756da2fc8910fe` |
| `test_encoder` | `True` | `1165512` | `c5bfb29b19599570fc6d39a3d9292414878534b4ff5fe4a10db1c5415630f6c0` |
| `test_adamw` | `True` | `1143208` | `207cdd2756cd520d44dca4adf693e18e5f77eec5d83780739b73936722cf9f38` |
| `test_global_norm` | `True` | `1138880` | `06c214a9bcc8dfa4b16177d3b3d7a3af7f34b61698083ab88b77a2bfe0c8c641` |
| `bench_sm120_matmul` | `True` | `2373912` | `385b0aa97efb919b3e2c6db0bd6130a192d6b7bc08af97cde8c41ddad2fdb1b9` |
| `bench_sm120_attention` | `True` | `1731352` | `596e98496c7b28e217d5072c25aa6ee8f9134a28bcc3e0d202a6347211ec65b0` |
| `bench_sm120_layernorm` | `True` | `1233728` | `423eee92bd9731920025739851005055d7dd32b38e1849afd1644cea7320e027` |
| `bench_sm120_runtime` | `True` | `2207936` | `7f4d38bb2c3b8b0c78b708743b9ffb3b28fe2383376909778d91217cc4e913d4` |
| `train_gpt2cu` | `True` | `3054888` | `631a0c3e1e7e8605b9cb725d6858bbf63916e3c3a46cd37ffab2e2bc66d82412` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
