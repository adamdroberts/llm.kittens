# SM120 Round Manifest

- run label: `codex_sm120_promoted_memory_store_cg_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_memory_store_cg_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_memory_store_cg_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `620`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `e3566cab3c718e5b99285e8087c3f63d06566913c8e304d441a8e8abf7466244` |
| `test_attention` | `True` | `1800528` | `ba70eb816de4aadafff3b1f317649432bec3965a12a4b4e0c6b9c4af2315d865` |
| `test_layernorm` | `True` | `1278296` | `4c9cbc5db6fccf50a315f656a5ca679267242bcedcc6ef40364d8434bb82ece7` |
| `test_bias` | `True` | `2089120` | `58e94d27c7764c093b76fe8afd2f275f55c6e2f8e53287998e3e4353f07f2856` |
| `test_gelu` | `True` | `1179912` | `789d010f087a8990af53c0bd3f458795fe36c032023eff2cce703c21bb93a95a` |
| `test_fused_classifier` | `True` | `1208704` | `3a662cd89ec23634e34673b4e658eec459b9445e4f11014b7d3cdf1db2e4523b` |
| `test_encoder` | `True` | `1210168` | `597baf719da863f8c0cd64bfa228691f932485db4fc5ba8ac6064670d3bce6a0` |
| `test_adamw` | `True` | `1183768` | `0d4bb9a453795a80e3f84c207b591f38be690a2333c5599fc083ee53d7a6fe99` |
| `test_global_norm` | `True` | `1179464` | `2227b098b8a7d91c4a9e8c7570cdfe586fb618ee7689db76682339351581501b` |
| `bench_sm120_matmul` | `True` | `2410304` | `6bc06855989f6dd239182e13e06929fa0266359de661ed8b0a75b20f13867728` |
| `bench_sm120_attention` | `True` | `1768800` | `0c5fef05af5e4ce3a0072ae6ba46d804d42b84182c82c4d8095b799bb7725d1c` |
| `bench_sm120_layernorm` | `True` | `1274232` | `3b08f92080885c38978537c6cd1c68418275bf10ce08ba7fe5cff51831959d87` |
| `bench_sm120_runtime` | `True` | `2271168` | `02f3f8b9cfb3331143ab15bc66892981f5af2d5e783a2ccc324bd4430014d64f` |
| `train_gpt2cu` | `True` | `3105552` | `c02b8a62028cb85f8880220918d3682ebc1798be0d5622f93aee878983445a52` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
