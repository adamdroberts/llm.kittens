# SM120 Round Manifest

- run label: `codex_sm120_round_layernorm_forward_nosmem_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_layernorm_forward_nosmem_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_layernorm_forward_nosmem_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `453`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2128288` | `f1a453c6e9eedcb78fe2f5a809035599d1a164c5065cb2539b9d2e7f54f063e5` |
| `test_attention` | `True` | `1760032` | `e8794504b8a7de2bd6c49131962b2939f4debeea4380e30bfeac6e083641d62d` |
| `test_layernorm` | `True` | `1237784` | `28181bf2253a986e4b1e519d4d11f42c8dff9a17b5f1e734b6d6b20183596090` |
| `test_bias` | `True` | `2039504` | `1344f9dd07f20bbeccebbccd4f1067c28b38a4ccca1b449c9e9125c1b2a4bf5d` |
| `test_gelu` | `True` | `1139336` | `acf481590a41458550b32464ea58ceb94d25fa87b95fbdac6ec417d732c9f925` |
| `test_fused_classifier` | `True` | `1146768` | `ea9e71d1e3a1092998782e8468b997f7c37bc6295b0267244b5fc2cad035511a` |
| `test_encoder` | `True` | `1165512` | `003a9e31ee9bc70bb866cb7c9c6d11bfc50e67d54d7f3f21c14f971974f7b14b` |
| `test_adamw` | `True` | `1138408` | `0eaae59a14191765e15a31735776e6105125eb19318b4adbeef3e7f1cf387bc2` |
| `test_global_norm` | `True` | `1138816` | `99b58ea198b2190d967e1126a6994200d5841c5bb8851d42b32837021c9571b1` |
| `bench_sm120_matmul` | `True` | `2365064` | `1c2971c189ec6fe6753852de524dc48bbf1ab0a1da760cdcc3e1a434ef3112c2` |
| `bench_sm120_attention` | `True` | `1731352` | `1a3f0366473613544589d24e9279d9dd2b9fcb778a15bc323837734a6fda4da3` |
| `bench_sm120_layernorm` | `True` | `1233536` | `64b3dfe9af4ade39aee6eb925388e915dc457cefbbe68de3401610d74a914e49` |
| `bench_sm120_runtime` | `True` | `2168552` | `c775bd2547d388224c098eeb4841f57cf7aeb3041f138dff0a5319d3e446f058` |
| `train_gpt2cu` | `True` | `3037016` | `7b8d7d8fc120b5c05f16e124f97a11ee2656e56b9dea2b8f365d16bf70cba552` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
