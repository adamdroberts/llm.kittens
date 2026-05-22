# SM120 Round Manifest

- run label: `codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block256_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `601`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `cf70f1f57c5039fe5ec06a09472fa6ff0b05a4744321d76be0dbb2c1e4db9b3c` |
| `test_attention` | `True` | `1800528` | `280890ab14a61bd8521d8e219400d5704dbd478ed6deb65cb53da658ca7e09ef` |
| `test_layernorm` | `True` | `1278296` | `d6758303645f9808e77505221edffa2b3276d21f3026efbf3d98c2bc45252965` |
| `test_bias` | `True` | `2089120` | `dec27028e1d0a9a1a8ad84024daac7c1b53aa7e5501b686134f3a3582b987613` |
| `test_gelu` | `True` | `1179912` | `f0675a105337b0f25b6e15a4540f34f1b9064f333a952cc83d837daec5aa8026` |
| `test_fused_classifier` | `True` | `1208704` | `9b069eabedc89ae9399f5c7053caf94e1cb498e8b837c33412cb4b0efaae46cc` |
| `test_encoder` | `True` | `1210168` | `10671620c581329d5dc2a30bf6523d3283c4f150d3feb3ff70614e5c6f2613c6` |
| `test_adamw` | `True` | `1183768` | `a1328b877d826dd9ebb1b7d00bbcf1db435df837070adc6154683ae55d7c4752` |
| `test_global_norm` | `True` | `1179464` | `f4dacbe38d78c76db2a60f9220604fd22dd5f70cf4ec9e568e4ea1ab290f80ff` |
| `bench_sm120_matmul` | `True` | `2410304` | `af58b086b945c06d4adc64e7c80bc99b8c91fd501b519c74bb5d51f0c5aa385e` |
| `bench_sm120_attention` | `True` | `1768800` | `85c57849ef41ea4c967fce997b2cd33a0b39af32c8dd0cfb4d404a47dd5c7dca` |
| `bench_sm120_layernorm` | `True` | `1274232` | `01c31980686c71f3a7a9458491b24be6e0a3e9dd7471c79f1df0f9da6ece2c11` |
| `bench_sm120_runtime` | `True` | `2271168` | `a6fa5f5b8f7b64258d0415ab41b242196277df94fa6f5b395a5fdf2ddae87eaf` |
| `train_gpt2cu` | `True` | `3105552` | `4dd1d7a44f594b937c30badacc64290dc2dca2c3e3059e98cc2fa9926a6c271b` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
