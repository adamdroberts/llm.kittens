# SM120 Round Manifest

- run label: `codex_sm120_round_adamw1024_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_adamw1024_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_adamw1024_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `438`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2115600` | `e1347cc825f2eca0b6309f116936a08f5bba63968a8c8f395d3313cdad1bb29c` |
| `test_attention` | `True` | `1760032` | `d4b42d635cfbf5a3d4d87b9521d7b62618f98f75774e82934bc5f39ca2d857ce` |
| `test_layernorm` | `True` | `1237784` | `f3ee7112b50df28d7a62ca73dab8f6f95c3a8513c08f2514b14da15ff26a498e` |
| `test_bias` | `True` | `2039504` | `234a592a291061025c7c1a7beacfd553ef10a40ecc7927467b0a41262b0e09db` |
| `test_gelu` | `True` | `1139336` | `45eb2216e004dcd0f269506dc126f037f60278395831f90656c449436defcefa` |
| `test_fused_classifier` | `True` | `1146768` | `a862eec005be69c1eefe827b469d017305d7a223b6e0053de663238e8e6aa095` |
| `test_encoder` | `True` | `1165512` | `ee99749fca280ea829c28167f45ac351b1a2184bb7eb6c4715b597906e02e582` |
| `test_adamw` | `True` | `1143216` | `9b9d61b27b7b26eb969a378430f1451e22606a2b72b23b4a590e0df3604e404a` |
| `test_global_norm` | `True` | `1138816` | `2b460ec694bdc848c6337c931fc9b8341538d2fdeb10055ceb5dd378549c1fc1` |
| `bench_sm120_matmul` | `True` | `2344376` | `2cf1a7c6c680f3b5d8d0ca7adfd12d39f6e5504cc08a749b219aa854f6c061ab` |
| `bench_sm120_attention` | `True` | `1731352` | `76dd264aa7ba101d9822909a2ee9fd9443de7cf2ede120e75e5769342ccffbed` |
| `bench_sm120_layernorm` | `True` | `1229088` | `4f22b756df5158009c812b20ca00c91e649bf1e217f559ad45a438e602ab895d` |
| `bench_sm120_runtime` | `True` | `2173376` | `fba6e28bd5364b196d0bdc580c1558dbc9ccf3b2c80238222b83598e29ac712d` |
| `train_gpt2cu` | `True` | `3041208` | `4d40ae79403aac846c0f75457f3d3f4cf957493250b2bc556a8d314e15825ebe` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
