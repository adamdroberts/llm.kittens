# SM120 Round Manifest

- run label: `codex_sm120_fcproj_precompute_zero0_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_fcproj_precompute_zero0_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_fcproj_precompute_zero0_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `671`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `e6e64508c883c502925914b23126b08dd4eaac345f1f59bc4df27ea53fd2dbf5` |
| `test_attention` | `True` | `1800528` | `142d2b59d37f280d5995f24111bf381a200c582530560ba7bf90232fd88928d7` |
| `test_layernorm` | `True` | `1278296` | `218885b6c7e15ac1ad4022e90991c0f4b2c2783de739d068f69a71c6acbd8ad0` |
| `test_bias` | `True` | `2089120` | `b96b6d275ddb0e2620316f4d2f1910a9aff100c4074f51b660138815d7e3e304` |
| `test_gelu` | `True` | `1179912` | `4a421538a9df1575269967e0d24a77d06da4796dd6a86e4d5ed1bf66d5b6499b` |
| `test_fused_classifier` | `True` | `1208704` | `9dc8cda29cd0c02535c14d2fe1500efaf0abe4054b9be3a82350bbbf4e4c7744` |
| `test_encoder` | `True` | `1210168` | `1ebff57dbe6a25a6894413ff42738b41705eaa6a63b45ec0d5f9cef28fea84b2` |
| `test_adamw` | `True` | `1183768` | `d7f63cb052aa2345b04a2ddb7d3de5cc57181826f679b42ce42b9683d81c6a62` |
| `test_global_norm` | `True` | `1179464` | `220cf90787a87ca301206e12a9c981d0c92d3643fbdc35afb615e3f1d01eb67c` |
| `bench_sm120_matmul` | `True` | `2410304` | `32db44b74130a1c87dbcfeaa9c1d54077cb6ba1b501f89aaf5e343e126d41752` |
| `bench_sm120_attention` | `True` | `1768800` | `02378bc6f076ccc7c0531639ad5f2219b527aaabd34e398ef8ca3ade3614e972` |
| `bench_sm120_layernorm` | `True` | `1274232` | `bcff71091e47489351b6d16d92bd9f6a43882e1953346e4c1161d9a6bb131663` |
| `bench_sm120_runtime` | `True` | `2271168` | `0e29436f4ee4c8884256919a5ddd3e7ea7b0e5bd1e5cb887ddd44e1c896ada8f` |
| `train_gpt2cu` | `True` | `3119216` | `301760416174a60975ac2f865c12fa5b664196714565b44defcb2ddce7ee44f0` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
