# SM120 Round Manifest

- run label: `codex_sm120_round_after_lnfix_clsfix_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_after_lnfix_clsfix_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_after_lnfix_clsfix_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `433`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2115600` | `29f1bb236d20aee68f00b5d5de9711a446dcd71d766cd6d03c6754824312659f` |
| `test_attention` | `True` | `1760032` | `fee1f6f727614117e58375cf699db78c77c8ffa5dc67da805626f2f00ab516c0` |
| `test_layernorm` | `True` | `1237784` | `a18571d82f16f3af73441a3c3432d38682130765cde9a45d802532b287a3ac41` |
| `test_bias` | `True` | `2039504` | `8922e18e88d92c0f39b2c0867b683d3c94a9e7862cd4024cb2c02750c766e5f1` |
| `test_gelu` | `True` | `1139336` | `ab8bc8d4932cb83c78dafd3a335db6ae3f5bd496693da88f741d7b7c783e71be` |
| `test_fused_classifier` | `True` | `1146768` | `73b0846f183356dc110a5120a37df89226dba7c37ca2f4331ae277d6b0c7952c` |
| `test_encoder` | `True` | `1165512` | `ae493d67a128c548e5e13c89d00aa551be3a8670f01037af27fa5ba3be778cb1` |
| `test_adamw` | `True` | `1134168` | `2b34f3e88799b69585d2fb9a1249f785f9056d65f63cc6451a466956085e4efb` |
| `test_global_norm` | `True` | `1138816` | `b0b8c17ca8875de90e22798391f16aa1613b2ed89c1e0bcbdf51333b231af6f1` |
| `bench_sm120_matmul` | `True` | `2344376` | `6c6ff6e4fc3aa7d5fb615ee03a90ba98fc985d95c3dc564fdb13b2a2b26f886a` |
| `bench_sm120_attention` | `True` | `1731352` | `d176fac665474d7b71e7920d2bd06eecdba0d1a6b742133541ecef3fd910d9af` |
| `bench_sm120_layernorm` | `True` | `1229088` | `af0a010c9c8e96743217dd94c69147828cb5c2d76f76d4380daeb0edc9c3c040` |
| `bench_sm120_runtime` | `True` | `2168552` | `16de373c487d2e99f6242c2341daa4843a61d441f2a38d980991ef07ad6d5cd7` |
| `train_gpt2cu` | `True` | `3036536` | `5da7a25f2ae56455900baa8bd6f9d27fe9ca14acb5f831a6bd58cdb38e3bc2e5` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
