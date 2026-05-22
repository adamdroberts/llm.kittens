# SM120 Round Manifest

- run label: `codex_sm120_selected_current_audit_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_selected_current_audit_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_selected_current_audit_x10_20260522`
- device arch: `SM120`
- max steps: `10`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `684`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `e72e69c2fbc907d69e0b3b6260968bc8a055a812fe3653327095163b48fc4d2b` |
| `test_attention` | `True` | `1800528` | `b01f864063e7c20eab4483658288692e31dd3f06fcab8d7e7192efeb940bd3dc` |
| `test_layernorm` | `True` | `1278296` | `5805b1b0170ef43aad142e5cfa7c0d6fcf3208a7514b9ca852b9a9e4c12514da` |
| `test_bias` | `True` | `2089120` | `7df36bb25a14d57f06b0db5eefdd26131d5eb6daa93dab855ad552dbaaf99e69` |
| `test_gelu` | `True` | `1179912` | `24d6bd1a7ab6c523df3df4edda6956291a99c3fdf02e6ef45126162390839d92` |
| `test_fused_classifier` | `True` | `1208704` | `db3a42feabf62b5f67010e289323b3c4a21fb98991a89a018c2a8e1ee3391a6d` |
| `test_encoder` | `True` | `1210168` | `7a9a35643d34bfdc749a1602332e8c94d7ea17d55214ccc5263606b77ba91306` |
| `test_adamw` | `True` | `1183768` | `b76e68098b81db1baddc49ecb5fc39bd4715ed64917f25f77c64f9ae1220605e` |
| `test_global_norm` | `True` | `1179464` | `5d7e2e352c641bd351f662850181c0bfae69d76270a0da04909cf4acd7a5683f` |
| `bench_sm120_matmul` | `True` | `2410304` | `f601a77757ed00ffb7a8b09e981211202bd9811d970b4138505e9a10d6c20a78` |
| `bench_sm120_attention` | `True` | `1768800` | `4eaf578bb478b280d3819905df3b9dc2b4cc761c4f4fc0aca68127451007d377` |
| `bench_sm120_layernorm` | `True` | `1274232` | `8149d1e14bed576a3bfec47b5197ea7f5874201d305758f9439ee09cb9799519` |
| `bench_sm120_runtime` | `True` | `2271168` | `5072305da385a04b71a83ef8dca1494ebad7b6f6117ff7883b024543619f228e` |
| `train_gpt2cu` | `True` | `3105552` | `9d329e98b96bd9e2a9e2a5c7dd4686b04ddba5ae4c853c69e85583572954751c` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
