# SM120 Round Manifest

- run label: `codex_sm120_promoted_cublas_dinp_fcproj_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_fcproj_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_cublas_dinp_fcproj_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `626`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `0efe7d8e60313ebd57668d6ec881b116d3dd91ccd7e5e6eb9c7f30b0eca65c59` |
| `test_attention` | `True` | `1800528` | `074681344063c6a6a23ee37d195a4ea75931a5558a119af6b5975eaff639b590` |
| `test_layernorm` | `True` | `1278296` | `4613edba59ddfc872dba09abb3eeaa9b182379642d8a74f17de3a040f4fc2da2` |
| `test_bias` | `True` | `2089120` | `ca4a81723935e0eea994bd7ce28fffbc87d54cc9a1d61766a00b389cb2ac6e82` |
| `test_gelu` | `True` | `1179912` | `f1ce3659c86fc98f753b8267e43c689cef4bd3fcab4c3ca756a06da40f92ffe6` |
| `test_fused_classifier` | `True` | `1208704` | `d94400c46adc345a0778678f423b7bf7708e6e43ee69d28caa977656fc44a4ca` |
| `test_encoder` | `True` | `1210168` | `9b4d62910715253c3eb823c58ddd5775f7f5fbf09097e86e35a4c5b000fd3d5e` |
| `test_adamw` | `True` | `1183768` | `c3292ebbdd5a4b8e61341312ec1861221a59cd66a9c4e550e94d153f75ec2bfc` |
| `test_global_norm` | `True` | `1179464` | `e04b8e7fc7992fe14d641e1e0a7b43fa06c2645655c7dab2166e65d09c51ecaf` |
| `bench_sm120_matmul` | `True` | `2410304` | `54a12f8f93745bb6bf26e399bf9972593090405e54e14f14fe42b424501d937a` |
| `bench_sm120_attention` | `True` | `1768800` | `5a223e57b820ee39e96ae15192dde3c04676499470caa3e00494506c61b9bbe2` |
| `bench_sm120_layernorm` | `True` | `1274232` | `a25914f8aabcbfa25c8fa1407b9ae4076d9dcfac817cc8380376d2294573812a` |
| `bench_sm120_runtime` | `True` | `2271168` | `9d2e2c031f7296041fea539d63d7a8517b25884af24a24e949126114e93a84b5` |
| `train_gpt2cu` | `True` | `3105552` | `b19a5f6214ffe09c4e94f6f783f1b915f4f7f9589e42eda3a1a6706808357f65` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
