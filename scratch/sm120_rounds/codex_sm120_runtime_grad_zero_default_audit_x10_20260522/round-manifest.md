# SM120 Round Manifest

- run label: `codex_sm120_runtime_grad_zero_default_audit_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_runtime_grad_zero_default_audit_x10_20260522`
- device arch: `SM120`
- max steps: `10`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `688`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `8ee0f3b4a6a6565c7a2a01371443ed7a605e00b9fe7a0fab7956bd362dec1ad8` |
| `test_attention` | `True` | `1800528` | `5d6a9d8189ee6e25183a11c8e366eb8d5e5e3ad438f2fc9719bf52c2f5018a06` |
| `test_layernorm` | `True` | `1278296` | `26dd92b1b48a78310736adb7f038953b924428ec2241ecf050562575de045f0a` |
| `test_bias` | `True` | `2089120` | `22a1190235b7011ae880f25d3ca09a6aca8f93e1375d3e0fd0b4dc1ddee2f3b7` |
| `test_gelu` | `True` | `1179912` | `025f8efd78302e49644a67f556fb809b5cf261df83bd904430a8163836c107b5` |
| `test_fused_classifier` | `True` | `1208704` | `b02350c101e9e5f20792d0159315921201d0280238795f390db80e9695dfe8b2` |
| `test_encoder` | `True` | `1210168` | `caca1bb60dd0034f8798d820d3821f00fbe317f8fb162c0815622bb581d93816` |
| `test_adamw` | `True` | `1183768` | `a8266131c64f590b28b2be3fb4e9c0f7d8320acb55c81f54c014004fd9b4ad97` |
| `test_global_norm` | `True` | `1179464` | `33120b5085b9d1c33576d61eb97332aee0c3a14f441568fc8b8594afb8f6a7ce` |
| `bench_sm120_matmul` | `True` | `2410304` | `8c48ebc2e1172189bf1f6b5f81ff4ed8bba2023a151a79de46d1672dfec95aac` |
| `bench_sm120_attention` | `True` | `1768800` | `423986151c7c6ddd50b6495d61f9c332977bf35fbb282e158b70518e51704921` |
| `bench_sm120_layernorm` | `True` | `1274232` | `881ff418bdc0226a8b24d6f6b1603030a41c78e95e9d2226d5e83c81b02725a0` |
| `bench_sm120_runtime` | `True` | `2271168` | `8c5b0277e14ad86a900c7627b6490f7937defc5e98c77ee42de7b334ff3b8753` |
| `train_gpt2cu` | `True` | `3105552` | `4bfe515d8f36aee2a88b63a6fb0229469eca50ce338975ae363ad06bcca99f1c` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
